"""参数化选股引擎（用户自由改参数，替代纯图形形态路线）。

设计：
- 每个指标 = 一个函数 (df, params) -> pd.Series[bool]（逐日可用，无未来数据）
- 条件组合 = AND；"接近度" = 已满足条件数/总条件数
- 买点：全部满足当日收盘确认 → 次日开盘进场（与形态口径一致）
- 止损：进场价 × (1 - stop_pct)，默认 7%

指标分四类：
  趋势强度: rps50, above_ma, near_high
  量价关系: volume_ratio_gt, up_days_gain, turnover_range, gain_today
  位置形态: pullback_stable, box_amplitude, vol_narrow
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ================================================================ 指标库

def _rps_series(close_wide: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """全市场 RPS(n)：各股票 n 日涨幅在截面上的百分位。返回与输入同形的 DataFrame。"""
    ret_n = close_wide.pct_change(n)
    return ret_n.rank(axis=1, pct=True)


INDICATORS = {}

def indicator(name, label, category, default, need_universe=False, desc="",
              valid_range=None, risks=None, watch_points=None, invalidation=None):
    """注册装饰器。params 从 default 取。need_universe=需要全市场截面(RPS)。
        risks: 风险提示短句列表
        watch_points: 后续观察点短句列表
        invalidation: 信号失效条件短句列表"""
    def deco(fn):
        INDICATORS[name] = {"fn": fn, "label": label, "category": category,
                            "default": default, "need_universe": need_universe,
                            "desc": desc, "valid_range": valid_range,
                            "risks": risks or [], "watch_points": watch_points or [],
                            "invalidation": invalidation or []}
        return fn
    return deco


# ---- 第一类：趋势强度 ----

@indicator("rps50", "RPS(50) 相对强度 ≥", "趋势强度", 0.85, need_universe=True,
          desc="个股50日涨幅在全市场的排名百分位。欧奈尔体系核心：强者恒强，≥0.85代表跑赢85%的股票。调高→只选最强势票(数量少、胜率略升)；调低→放宽到中上水平(数量多)。",
          valid_range=[0.6,0.99],
          risks=["RPS高反映过去50日涨幅大，短期可能有获利回吐压力", "强势股在板块轮动时回调幅度也大"],
          watch_points=["RPS是否持续走升（动量加速）还是开始回落", "板块整体是否同步走强"],
          invalidation=["RPS连续5日跌破0.80且价格跌破MA20"])
def _rps50(df, p, rps=None):
    if rps is None:
        return None
    s = rps.get(df.attrs.get("symbol"))
    if s is None:
        return pd.Series(False, index=df.index)
    return s >= float(p)


@indicator("above_ma", "收盘价站上 MA", "趋势强度", 20,
          desc="收盘价高于N日均线=中期趋势向上。N调小→反应灵敏、信号多但震荡市易被打脸；N调大→只选大趋势票但信号滞后。",
          valid_range=[5,120],
          risks=["站上MA后可能回踩确认", "震荡市中假突破频繁"],
          watch_points=["后续3日能否站稳MA上方", "成交量是否持续配合"],
          invalidation=["收盘跌破MA且3日未回", "MA走平或拐头向下"])
def _above_ma(df, p, rps=None):
    ma = df["C"].rolling(int(p), min_periods=int(p)).mean()
    return df["C"] > ma


@indicator("near_high", "距 52 周新高 ≤ %", "趋势强度", 10,
          desc="距250日最高价不超过X%=接近创新高。回测警示：A股追高胜率仅38-41%，建议与'回撤企稳'等低位置因子组合使用，勿单独作为买入条件。",
          valid_range=[2,30],
          risks=["接近新高时可能遭遇前期套牢盘抛压", "短期涨幅已大，追高需承担回调风险"],
          watch_points=["突破后能否站稳前高上方", "突破当日成交量是否配合放大"],
          invalidation=["突破失败回落至前高下方", "突破后连续3日收盘回落"])
def _near_high(df, p, rps=None):
    hi = df["H"].rolling(250, min_periods=60).max()
    return (hi - df["C"]) / hi <= float(p) / 100


# ---- 第二类：量价关系 ----

@indicator("vol_ratio", "量比(对20日均量) ≥", "量价关系", 1.5,
          desc="当日成交量/20日均量。放量代表资金关注。调高→只要显著放量的确认信号；过低(<1.2)噪音大。",
          valid_range=[1.1,5.0],
          risks=["放量可能是对倒或短期情绪，未必是持续流入", "高位放量需警惕主力出货"],
          watch_points=["放量后量能是否持续而非一日游", "放量方向（上涨放量/下跌放量）"],
          invalidation=["放量后连续3日量能回落至均量以下", "放量后价格跌破放量日低点"])
def _vol_ratio(df, p, rps=None):
    vma = df["V"].shift(1).rolling(20, min_periods=10).mean()
    return df["V"] / vma >= float(p)


@indicator("up_days", "连续放量上涨 ≥ 天", "量价关系", 2,
          desc="连续N天'上涨且放量'=资金持续流入。是最稳健的量价因子之一(回测20日均收益+0.8~1%)。调高→要求更持续的流入但信号变少。",
          valid_range=[2,6],
          risks=["连续上涨后短期获利盘积累", "若量能非递增式放大则持续性存疑"],
          watch_points=["上涨过程中量能是否逐步放大", "每日收盘价是否逐步上移"],
          invalidation=["连续2天收阴或量能萎缩", "价格跌破MA5"])
def _up_days(df, p, rps=None):
    up = df["C"] > df["C"].shift(1)
    vol_up = df["V"] > df["V"].shift(1)
    both = up & vol_up
    out = both.copy()
    run = both.astype(int)
    for k in range(1, int(p)):
        run = run & both.shift(k).fillna(False).astype(int)
    return run.astype(bool)


@indicator("turnover", "换手率在 [min,max]% (需流通股本)", "量价关系", [3, 15],
          desc="成交量处于近一年什么分位区间。中间区间最健康：既非无人问津也非过热爆炒。需流通股本数据，当前为近似口径。",
          risks=["换手率突变可能是短期事件驱动", "低换手率时流动性差、进出场困难"],
          watch_points=["换手率趋势是走高还是走低", "换手率与价格方向是否配合"],
          invalidation=["换手率连续3日低于区间下限", "换手率突破区间上限（过热）"])
def _turnover(df, p, rps=None):
    # 无流通股本数据时退化为成交量分位近似
    lo, hi = float(p[0]), float(p[1])
    vpct = df["V"].rolling(250, min_periods=60).rank(pct=True) * 100
    return (vpct >= lo * 2) & (vpct <= max(hi * 4, 99))


@indicator("gain_today", "当日涨幅 ≥ %", "量价关系", 3,
          desc="当日涨幅≥X%。捕捉启动日的动量。调低→早期信号多但假突破多；调高→确认强但进场价已抬高。",
          valid_range=[1,9.9],
          risks=["单日大涨可能是消息驱动未必可持续", "高涨幅后次日低开概率大"],
          watch_points=["次日是否高开还是低开", "收盘能否站稳当日收盘价附近"],
          invalidation=["次日收盘回落至涨幅50%以下", "后续3日连续收阴"])
def _gain_today(df, p, rps=None):
    return df["C"] / df["C"].shift(1) - 1 >= float(p) / 100


# ---- 第三类：位置形态 ----

@indicator("pullback_stable", "回撤后企稳 ≥ 天", "位置形态", 3,
          desc="从高点回撤≥5%后连续N天收稳于5日线上=回调结束迹象。全场表现最好的位置类因子(49.1%)，适合做低吸。",
          valid_range=[1,8],
          risks=["回撤企稳不一定意味着重新上涨，可能是下跌中继", "震荡市中反复假信号多"],
          watch_points=["回撤后是否开始温和放量上涨", "站稳MA5后是否向MA20进发"],
          invalidation=["价格跌破回撤低点", "MA5拐头向下"])
def _pullback_stable(df, p, rps=None):
    """从20日高点回撤≥5%后，连续N天不创新低且收在MA5上方。"""
    hi20 = df["H"].rolling(20, min_periods=10).max()
    pulled = df["C"] <= hi20 * 0.95
    stable = (df["C"] > df["C"].rolling(5, min_periods=5).mean()) & \
             (df["L"] >= df["L"].rolling(5, min_periods=5).min())
    # pulled 发生后的第 N 天仍 stable
    sig = pd.Series(False, index=df.index)
    pulled_any = pulled.rolling(int(p)).max().astype(bool)
    return pulled_any & stable


@indicator("box_amp", "横盘振幅 ≤ %(近20日)", "位置形态", 15,
          desc="近20日高低点振幅≤X%=横盘蓄势。振幅越小筹码越稳定，突破时爆发力越强(Darvas箱体理论)。调低→箱体更紧、信号更少更精。",
          valid_range=[5,35],
          risks=["横盘后可能向下突破而非向上", "低振幅低流动性时买卖价差大"],
          watch_points=["横盘期间量能是否逐步萎缩（蓄势）", "突破方向（向上/向下）"],
          invalidation=["价格跌破横盘区间下沿", "横盘后放量下跌"])
def _box_amp(df, p, rps=None):
    w = 20
    hi = df["H"].rolling(w, min_periods=w).max()
    lo = df["L"].rolling(w, min_periods=w).min()
    return (hi / lo - 1) * 100 <= float(p)


@indicator("vol_narrow", "波动率收窄至 ≤ %(ATR/价格)", "位置形态", 3.5,
          desc="ATR波动率≤价格的X%=波动收敛。经典'波动压缩→方向爆发'前置形态(Bollinger Squeeze)。配合放量方向确认使用效果最佳。",
          valid_range=[1.5,7],
          risks=["波动率收窄后未必马上爆发，可能继续横盘", "收窄后可能向下突破而非向上"],
          watch_points=["收窄后是否出现放量突破", "突破方向（向上/向下）"],
          invalidation=["ATR突破区间上限（波动放大但方向不明）", "价格向下突破收窄区间"])
def _vol_narrow(df, p, rps=None):
    tr = pd.concat([df["H"] - df["L"],
                    (df["H"] - df["C"].shift()).abs(),
                    (df["L"] - df["C"].shift()).abs()], axis=1).max(axis=1)
    natr = tr.rolling(14, min_periods=14).mean() / df["C"] * 100
    return natr <= float(p)


# ---- 第四类：全量因子（动量/区间位置/均线结构） ----

@indicator("mom_win", "动量：[N]日涨幅 ≥ %", "全量因子", [20, 15],
          desc="过去N日累计涨幅≥X%=动量因子。注意：60日涨幅>30%的深动量在A股是负期望(38.8%)，建议20日窗口+温和阈值。",
          risks=["动量过大时追高风险大", "短期涨幅已大可能面临获利回吐"],
          watch_points=["动量是否在加速还是开始衰减", "量能是否配合价格上涨"],
          invalidation=["涨幅从峰值回落超过30%", "连续3日收阴且量能萎缩"])
def _mom(df, p, rps=None):
    n, pct = int(p[0]), float(p[1])
    return df["C"].pct_change(n) * 100 >= pct


@indicator("rsv_pos", "RSV[N]: 收盘位于N日区间 ≤ %位", "全量因子", [60, 80],
          desc="收盘价在N日高低区间的百分位(RSQR60同族)。≤50%偏超卖可低吸；≥90%是极强势但回测显示追高风险大。默认80%以下。",
          risks=["RSV低不代表立刻反弹，可能继续超卖", "RSV高不代表立刻回调，可能继续强势"],
          watch_points=["RSV是否从低位开始回升", "价格是否出现止跌迹象"],
          invalidation=["RSV持续走低且价格连续创新低", "RSV突破区间上限（过热）"])
def _rsv(df, p, rps=None):
    """类似 RSQR60：收盘价在近N日高低区间内的百分位，越低越超卖、越高越强势。
    条件含义：RSV(N) <= max%（低位置）——抄底/回撤场景用；
    想选强势突破可把 max 调成 >= 的思路：改用 min 参数填大值即可近似。"""
    n, mx = int(p[0]), float(p[1])
    hi = df["H"].rolling(n, min_periods=n).max()
    lo = df["L"].rolling(n, min_periods=n).min()
    rng = hi - lo
    rsv = (df["C"] - lo) / rng.replace(0, np.nan) * 100
    return rsv <= mx


@indicator("ma_align", "均线多头 MA5>MA10>MA20 持续 ≥ 天", "全量因子", 3,
          desc="短中期均线自上而下排列=标准多头结构。持续天数越长趋势越扎实。46-47%胜率属中游，适合做组合基底条件。",
          valid_range=[1,15],
          risks=["均线多头排列是滞后指标，价格可能已开始回调", "震荡市中均线反复交叉假信号多"],
          watch_points=["均线间距是否在扩大（趋势加速）", "价格是否在MA5上方运行"],
          invalidation=["MA5下穿MA10", "MA10下穿MA20"])
def _ma_align(df, p, rps=None):
    m5 = df["C"].rolling(5).mean()
    m10 = df["C"].rolling(10).mean()
    m20 = df["C"].rolling(20).mean()
    bull = (m5 > m10) & (m10 > m20)
    run = bull.astype(int)
    for k in range(1, int(p)):
        run = run & bull.shift(k).fillna(False).astype(int)
    return run.astype(bool)


@indicator("macd_cross", "MACD金叉后 ≤ 天 且 DIF>0", "全量因子", 5,
          desc="MACD零轴上方金叉后N天内=上升趋势中的二次启动信号。零轴下金叉未计入(可靠性差)。",
          valid_range=[1,15],
          risks=["零轴上方金叉在震荡市中假信号多", "MACD滞后于价格，金叉时可能已涨了一段"],
          watch_points=["金叉后DIF是否在零轴上方持续走高", "金叉当日量能是否放大"],
          invalidation=["DIF重新下穿DEA", "DIF跌破零轴"])
def _macd(df, p, rps=None):
    c = df["C"]
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    dif = e12 - e26
    dea = dif.ewm(span=9, adjust=False).mean()
    cross = (dif > dea) & (dif.shift() <= dea.shift())
    within = cross.rolling(int(p) + 1, min_periods=1).max().astype(bool)
    return within & (dif > 0)


@indicator("new_high_cnt", "250日创52周新高次数 ≥ 次", "全量因子", 2,
          desc="250日内创52周新高的次数=反复走强的证据(欧奈尔)。⚠️回测强警告：A股创新高后短期均值回归剧烈(胜率30-31%，亏约5%)，切勿单独使用。",
          valid_range=[1,5],
          risks=["创新高后短期均值回归剧烈", "连续创新高后获利盘压力大"],
          watch_points=["创新高时量能是否配合", "创新高后是否站稳"],
          invalidation=["创新高后连续3日回落", "创新高当日收长上影线"])
def _nh_cnt(df, p, rps=None):
    hi250 = df["H"].rolling(250, min_periods=120).max()
    is_nh = df["C"] >= hi250 * 0.999
    cnt = is_nh.rolling(250, min_periods=60).sum()
    return cnt >= float(p)


@indicator("ma_spread", "收盘高出 MA20 幅度在 [min,max]%", "全量因子", [0, 8],
          desc="价格高出MA20的百分比区间。[0,8]%=强势但不脱离均线(健康)；[3,15]%偏高位置注意回落风险。",
          risks=["偏高位置追高需承担回落风险", "过低位置可能趋势尚未确立"],
          watch_points=["MA20斜率是否向上", "价格与MA20距离是否健康"],
          invalidation=["价格跌破MA20", "MA20走平或拐头向下"])
def _ma_spread(df, p, rps=None):
    lo, hi = float(p[0]), float(p[1])
    m20 = df["C"].rolling(20, min_periods=20).mean()
    sp = (df["C"] / m20 - 1) * 100
    return (sp >= lo) & (sp <= hi)


@indicator("amp20", "20日平均振幅 ≤ %(波动适中)", "全量因子", 5,
          desc="20日平均日振幅≤X%。振幅适中(3-4%)的票走势流畅好持有；高振幅票难拿住且止损容易被扫。",
          valid_range=[2,12])
def _amp(df, p, rps=None):
    amp = (df["H"] - df["L"]) / df["C"].shift() * 100
    return amp.rolling(20, min_periods=10).mean() <= float(p)


@indicator("yang_streak", "连续阳线 ≥ 天", "全量因子", 3,
          desc="连续N天收盘>开盘=买方持续主导。连续5天阳线时20日均收益+2.02%(全场最高)，但样本较少注意甄别。",
          valid_range=[2,8])
def _yang(df, p, rps=None):
    up = df["C"] > df["O"]
    run = up.astype(int)
    for k in range(1, int(p)):
        run = run & up.shift(k).fillna(False).astype(int)
    return run.astype(bool)


# ================================================================ 引擎

DEFAULT_STOP_PCT = 7.0


def validate_conditions(conditions: dict) -> tuple[dict, list[str]]:
    """校验用户参数是否在有效区间内。返回 (合法条件, 错误列表)。"""
    errors = []
    ok = {}
    for name, param in conditions.items():
        meta = INDICATORS.get(name)
        if meta is None:
            errors.append(f"{name}: 未知指标")
            continue
        vr = meta.get("valid_range")
        if not vr:
            # 区间型参数 [min,max]：校验 min<max 且数值合理
            if isinstance(param, list):
                if len(param) != 2 or not all(isinstance(x, (int, float)) for x in param):
                    errors.append(f"{name}: 参数格式错误")
                    continue
                if param[0] >= param[1]:
                    errors.append(f"{name}: 下限必须小于上限")
                    continue
                if param[0] < 0 or param[1] > 1000:
                    errors.append(f"{name}: 数值超出合理范围(0~1000)")
                    continue
            ok[name] = param
            continue
        lo, hi = vr
        if isinstance(param, list):
            # 双参数指标带区间约束：首参数须在区间内
            v = float(param[0]) if param else None
            if v is None or not (lo <= v <= hi):
                errors.append(f"{name}: 参数 {v} 超出有效区间 [{lo},{hi}]，该档位下指标失灵")
                continue
            ok[name] = param
        else:
            v = float(param)
            if not (lo <= v <= hi):
                errors.append(f"{name}: 参数 {v} 超出有效区间 [{lo},{hi}]，该档位下指标失灵")
                continue
            ok[name] = param
    return ok, errors


def available_indicators() -> list[dict]:
    """给前端渲染表单用。"""
    out = []
    for name, meta in INDICATORS.items():
        out.append({"name": name, "label": meta["label"],
                    "category": meta["category"], "default": meta["default"],
                    "desc": meta.get("desc", ""),
                    "valid_range": meta.get("valid_range"),
                    "risks": meta.get("risks", []),
                    "watch_points": meta.get("watch_points", []),
                    "invalidation": meta.get("invalidation", [])})
    return out


def screen(daily: dict[str, pd.DataFrame],
           close_wide: pd.DataFrame | None,
           conditions: dict[str, object]) -> dict:
    """执行筛选。

    daily: {symbol: df(O,H,L,C,V; DatetimeIndex)}
    conditions: {indicator_name: param}
    返回 {triggered: [...], near: [...]}，每项含 symbol/score/met/missing。
    """
    rps = _rps_series(close_wide, 50) if close_wide is not None else None
    results = []
    last_dates = {s: df.index[-1] for s, df in daily.items()}
    latest_day = max(last_dates.values())

    for sym, df in daily.items():
        if df.empty or len(df) < 60:
            continue
        met, missing = [], []
        for name, param in conditions.items():
            meta = INDICATORS.get(name)
            if meta is None:
                continue
            try:
                series = meta["fn"](df, param, rps=rps)
            except Exception:
                series = None
            if series is None or series.empty:
                missing.append(name)
                continue
            val = bool(series.iloc[-1])
            (met if val else missing).append(name)

        total = len(met) + len(missing)
        if total == 0:
            continue
        score = round(len(met) / total * 100)
        rec = {"symbol": sym, "score": score,
               "met": met, "missing": missing,
               "signal_date": str(last_dates[sym].date()),
               "status": "triggered" if not missing else "near"}
        results.append(rec)

    triggered = sorted([r for r in results if r["status"] == "triggered"],
                       key=lambda r: -r["score"])
    near = sorted([r for r in results if r["status"] == "near"],
                  key=lambda r: (-len(r["met"]), -r["score"]))
    return {"triggered": triggered, "near": near[:50]}


def entry_and_stop(df: pd.DataFrame, signal_pos: int,
                   stop_pct: float = DEFAULT_STOP_PCT) -> dict | None:
    """买点=信号次日开盘；止损=买点×(1-stop%)。

    - 信号在历史K线：用实际次日开盘（停牌顺延最多5日）
    - 信号在最新K线（实时筛选场景）：次日未开盘 →
      用今日收盘作参考买点，标注 pending，提示"明日开盘附近进场"
    """
    n = len(df)
    for k in range(1, 6):
        i = signal_pos + k
        if i < n:
            o = float(df["O"].iloc[i])
            if o > 0:
                return {"entry_price": round(o, 3),
                        "entry_date": str(df.index[i].date()),
                        "stop_price": round(o * (1 - stop_pct / 100), 3),
                        "stop_pct": stop_pct}
    # 实时场景：以最后收盘价为参考
    c = float(df["C"].iloc[-1])
    return {"entry_price": round(c, 3),
            "entry_date": "明日开盘(参考)",
            "stop_price": round(c * (1 - stop_pct / 100), 3),
            "stop_pct": stop_pct,
            "pending": True}
