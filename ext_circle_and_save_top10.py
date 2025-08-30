#######################################
#  Takako Yamada 20250830 version
#外周円とその中心座標がtest_matrix.csvで入っている．
#この外周円の内側のランドルド環を最大５つまで抽出するプログラム
#必要なのは，imagesというフォルダ内の静止画
#この静止画の中で，推定された外周円(青）と正解（緑）の差分が
#少なく，正解に近い推定された静止画10枚について，
#ランドルト環を推定し，空き方向を時刻（長針）で表示したものが，
#picked_top10フォルダ内にそれぞれ画像ファイル名をフォルダとして記録
#するプログラム，起動方法は
#python extract_circles1.py
########################################
#
#source ~/tfenv/bin/activate python
import matplotlib as mpl
import numpy as np
import math
import os
import re
import pandas as pd
import shutil
from pathlib import Path

# ---- Matplotlib backend 選択（保存が主目的なので最後は 'Agg' にフォールバック）----
for b in ("TkAgg", "Qt5Agg", "MacOSX", "Agg"):
    try:
        mpl.use(b, force=True)
        break
    except Exception:
        pass

print("Matplotlib backend:", mpl.get_backend())  # デバッグ表示

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Arc


# =====================
# 設定
# =====================
IN_CSV    = "test_metrics.csv"
OUT_CSV   = "test_metrics_filename.csv"  # path列をファイル名化した全件の保存先
IMG_DIR   = "images"                     # 画像探索ルート
TOP_K     = 10                           # 最小誤差で抽出する件数
OUT_ROOT  = "picked_top10"               # 出力ルートフォルダ
# 欠け検出パラメータ（必要に応じて調整）
GAP_PARAMS = dict(
    ring_is_bright="auto",
    n_samples=720,
    smooth_deg=6.0,
    k_std=0.6,
    min_gap_deg=8.0,
    band_frac=0.05,
    band_radial_samples=9,
    bg_probe_frac=0.01,
    bg_gap_px=2.0,
    max_rings=5,               # ★追加
    scan_inner_min_frac=0.15,  # 必要なら調整
    n_scan=28,
    min_rad_sep_frac=0.6,
    scale=2.0
)


# =====================
# ユーティリティ
# =====================
def basename_series(path_series: pd.Series) -> pd.Series:
    """パス文字列列からファイル名だけを安全に取り出す（区切りを/に正規化）"""
    p = (
        path_series.astype(str)
        .str.strip()
        .str.lstrip("\ufeff")           # 先頭BOM対策
        .str.replace("\\", "/", regex=False)
    )
    filenames = p.str.extract(r'([^/]+?\.(?:jpg|png))$', flags=re.IGNORECASE)[0]
    return filenames.fillna(p.str.split("/").str[-1])

def readcsv() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if "path" not in df.columns:
        raise KeyError("CSVに 'path' 列がありません。")
    df["path"] = basename_series(df["path"])
    df["path"].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"done -> {OUT_CSV}")
    return df

def pick_col(df: pd.DataFrame, *candidates) -> str:
    """候補名のうち存在する最初の列名を返す。見つからなければ空文字。"""
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def find_image(images_dir: str, query: str) -> str:
    """
    images_dir配下を再帰探索して、ファイル名に query を含む最有力候補を1件返す。
    見つからなければ images_dir/query を返す（存在有無は呼び出し側で確認）。
    """
    q = os.path.splitext(query)[0].casefold()
    hits = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            name = os.path.splitext(f)[0].casefold()
            if q in name:
                hits.append(os.path.join(root, f))
    if not hits:
        q_full = query.casefold()
        for root, _, files in os.walk(images_dir):
            for f in files:
                if q_full in f.casefold():
                    hits.append(os.path.join(root, f))
    if hits:
        EXT_ORDER = {".png": 0, ".jpg": 1, ".jpeg": 2, ".bmp": 3, ".tif": 4, ".tiff": 5}
        def rank(p: str):
            base = os.path.basename(p)
            stem, ext = os.path.splitext(base)
            return (len(stem), EXT_ORDER.get(ext.lower(), 9), base.casefold(), p.casefold())
        hits.sort(key=rank)
        return hits[0]
    return os.path.join(images_dir, query)

def getf(row, key):
    v = row.get(key, None)
    try:
        return float(v)
    except Exception:
        return None

# -------- 画像統計・サンプリング --------
def image_means(img):
    arr = img.astype(np.float32)
    if arr.max() > 1.5:  # 0-255なら0-1へ
        arr = arr / 255.0
    if arr.ndim == 2:
        gray = arr
        meanR = meanG = meanB = float(gray.mean())
    else:
        if arr.shape[2] >= 3:
            R, G, B = arr[...,0], arr[...,1], arr[...,2]
        else:
            R = G = B = arr[...,0]
        meanR, meanG, meanB = float(R.mean()), float(G.mean()), float(B.mean())
        gray = 0.2989*R + 0.5870*G + 0.1140*B
    return {"R_mean": meanR, "G_mean": meanG, "B_mean": meanB, "gray_mean": float(gray.mean()), "gray": gray}

def bilinear_sample_gray(gray, xs, ys):
    H, W = gray.shape
    x0 = np.clip(np.floor(xs).astype(int), 0, W-1)
    y0 = np.clip(np.floor(ys).astype(int), 0, H-1)
    x1 = np.clip(x0 + 1, 0, W-1)
    y1 = np.clip(y0 + 1, 0, H-1)
    wa = (x1 - xs) * (y1 - ys); wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0); wd = (xs - x0) * (ys - y0)
    Ia = gray[y0, x0]; Ib = gray[y0, x1]; Ic = gray[y1, x0]; Id = gray[y1, x1]
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def circular_moving_average(x, win_deg, n_samples):
    w = max(1, int(round(n_samples * win_deg / 360.0)))
    if w <= 1:
        return x.copy()
    y = np.zeros_like(x, dtype=np.float32)
    half = w // 2
    for k in range(-half, half+1):
        y += np.roll(x, k)
    y /= (2*half + 1)
    return y

def ring_intensity_profile(gray, cx, cy, r, n_samples=720):
    theta = np.linspace(0.0, 2*np.pi, n_samples, endpoint=False)
    xs = cx + r * np.cos(theta)
    ys = cy + r * np.sin(theta)  # 画像座標は下向きが+なので時計回り
    vals = bilinear_sample_gray(gray, xs, ys)
    theta_deg = (theta * 180.0 / np.pi) % 360.0
    return theta_deg, vals

def detect_gaps_on_ring(vals, ring_is_bright=True, smooth_deg=6.0, k_std=0.5, min_gap_deg=10.0):
    n = len(vals)
    sm = circular_moving_average(vals, smooth_deg, n)
    m, s = float(sm.mean()), float(sm.std() + 1e-6)
    if ring_is_bright:
        thr = m - k_std * s
        gap = sm < thr
    else:
        thr = m + k_std * s
        gap = sm > thr

    gap_ext = np.r_[False, gap, False]
    changes = np.flatnonzero(gap_ext[1:] != gap_ext[:-1])
    starts = changes[0::2]; ends = changes[1::2]
    to_deg = lambda idx: (idx / n) * 360.0
    intervals = []
    for st, ed in zip(starts, ends):
        length_deg = to_deg(ed - st)
        if length_deg >= min_gap_deg:
            intervals.append((to_deg(st), to_deg(ed)))
    return intervals, dict(mean=m, std=s, thr=thr)

def find_concentric_radii(pr_r, n_inner=4, inner_min=0.2, inner_max=0.85):
    """外側(=大)→内側(=小)の順に並べた半径を返す"""
    rs = np.linspace(inner_min*pr_r, inner_max*pr_r, n_inner)
    return rs[::-1]  # 外側→内側

# -------- 表示/保存系 --------
def save_overlay_with_circles(img_path: str, row, out_path: str, title: str = ""):
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title or os.path.basename(img_path))
    ax.axis("off")
    ax.set_aspect("equal")

    def getf_local(key):
        v = row.get(key, None)
        try:
            return float(v)
        except Exception:
            return None

    # gt or gr を吸収
    gt_x = getf_local("gt_x") if row.get("gt_x") is not None else getf_local("gr_x")
    gt_y = getf_local("gt_y") if row.get("gt_y") is not None else getf_local("gr_y")
    gt_r = getf_local("gt_r") if row.get("gt_r") is not None else getf_local("gr_r")

    pr_x = getf_local("pr_x"); pr_y = getf_local("pr_y"); pr_r = getf_local("pr_r")

    if None not in (gt_x, gt_y, gt_r):
        ax.add_patch(Circle((gt_x, gt_y), gt_r, fill=False, linewidth=2, edgecolor="g", zorder=3))
        ax.plot([gt_x], [gt_y], "o", markersize=4, color="g", zorder=4)
    if None not in (pr_x, pr_y, pr_r):
        ax.add_patch(Circle((pr_x, pr_y), pr_r, fill=False, linewidth=2, edgecolor="b", zorder=3))
        ax.plot([pr_x], [pr_y], "o", markersize=4, color="b", zorder=4)

    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def annulus_mean_gray(gray, cx, cy, r_inner, r_outer, n_theta=720, n_r=5):
    """
    中心(cx,cy)、半径[r_inner, r_outer] の「輪帯」の平均グレイ値を返す。
    角度n_theta×半径n_rで等間隔サンプルして平均。
    """
    H, W = gray.shape
    r_inner = max(0.0, float(r_inner))
    r_outer = max(r_inner, float(r_outer))
    if r_outer <= 1e-3:
        return float(gray.mean())

    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    rs     = np.linspace(r_inner, r_outer, max(1, int(n_r)))
    acc = 0.0; cnt = 0
    for r in rs:
        xs = cx + r * np.cos(thetas)
        ys = cy + r * np.sin(thetas)
        vals = bilinear_sample_gray(gray, xs, ys)
        acc += float(vals.mean()); cnt += 1
    return acc / max(1, cnt)

def band_intensity_profile(gray, cx, cy, r_center, band_px, n_radial=9, n_samples=720):
    """
    半径 r_center の周囲 band_px の“厚み”を持つ輪帯で、角度方向プロファイルを作る。
    （半径方向に n_radial 本サンプルして平均）
    戻り: theta_deg(0..360), vals(角度に対応する平均強度)
    """
    H, W = gray.shape
    band_px = float(band_px)
    if band_px <= 0:
        return ring_intensity_profile(gray, cx, cy, r_center, n_samples)

    r_in  = max(0.0, r_center - band_px/2.0)
    r_out = max(r_in + 1e-3, r_center + band_px/2.0)

    theta = np.linspace(0.0, 2*np.pi, n_samples, endpoint=False)
    vals_acc = np.zeros(n_samples, dtype=np.float32)

    rs = np.linspace(r_in, r_out, max(1, int(n_radial)))
    for r in rs:
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        vals_acc += bilinear_sample_gray(gray, xs, ys).astype(np.float32)

    vals = vals_acc / max(1, len(rs))
    theta_deg = (theta * 180.0 / np.pi) % 360.0
    return theta_deg, vals


#内側のみ・最大5本



# =====================
# ランキングして上位K件を処理
# =====================
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    df = readcsv()

    # gt_* と gr_* を吸収
    gx = pick_col(df, "gt_x", "gr_x")
    gy = pick_col(df, "gt_y", "gr_y")
    gr = pick_col(df, "gt_r", "gr_r")
    px = pick_col(df, "pr_x"); py = pick_col(df, "pr_y"); pr = pick_col(df, "pr_r")

    if not (gx and gy and gr and px and py and pr):
        missing = [("gt_x|gr_x", gx), ("gt_y|gr_y", gy), ("gt_r|gr_r", gr), ("pr_x", px), ("pr_y", py), ("pr_r", pr)]
        missing = [k for k, v in missing if not v]
        raise KeyError(f"必要列が不足しています: {missing}")

    # スコア（小さいほど良い）: center_err_px + radius_err_px を優先。なければ (dx,dy,dr) の合成。
    if "center_err_px" in df.columns and "radius_err_px" in df.columns:
        df["_rank_score"] = df["center_err_px"].astype(float).abs() + df["radius_err_px"].astype(float).abs()
    else:
        dx = df[gx].astype(float) - df[px].astype(float)
        dy = df[gy].astype(float) - df[py].astype(float)
        dr_ = df[gr].astype(float) - df[pr].astype(float)
        # 3次元のユークリッド距離（rもpx単位想定）
        df["_rank_score"] = np.hypot(np.hypot(dx, dy), dr_.abs())

    df_ranked = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[px, py, pr, gx, gy, gr, "_rank_score"])
    df_ranked = df_ranked.sort_values("_rank_score", ascending=True).head(TOP_K).reset_index(drop=True)

    print(f"選抜 {len(df_ranked)} 件（要求 {TOP_K} 件）")

    for idx, row in df_ranked.iterrows():
        base_name = os.path.basename(str(row["path"]))
        stem = os.path.splitext(base_name)[0]

        # 画像を検索
        img_path = find_image(IMG_DIR, base_name)
        if not os.path.isfile(img_path):
            print(f"[WARN] 画像が見つかりません: {img_path} -> スキップ")
            continue

        # 出力フォルダ（画像ファイル名）
        out_dir = Path(OUT_ROOT) / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 元画像をコピー
        copied_orig = out_dir / base_name
        try:
            if os.path.abspath(img_path) != os.path.abspath(str(copied_orig)):
                shutil.copy2(img_path, copied_orig)
        except Exception as e:
            print(f"[WARN] 元画像コピー失敗: {e}")

        # 円オーバーレイ画像（GT=緑, PR=青）
        overlay_path = out_dir / "overlay_circles.png"
        save_overlay_with_circles(img_path, row, str(overlay_path), title=f"{base_name} (rank #{idx+1})")

        # 欠け抽出画像
        pr_x = float(row[px]); pr_y = float(row[py]); pr_r = float(row[pr])
        gaps_img_path = out_dir / "gaps_debug.png"
                
        result = analyze_and_draw(
            image_path=img_path,
            pr_x=pr_x, pr_y=pr_y, pr_r=pr_r,
            out_path=str(gaps_img_path),
            try_show=False,
            **GAP_PARAMS
        )
        
        
        
        
        # summary.txt を作成
        summary_path = out_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# Summary for {base_name} (rank #{idx+1})\n")
            f.write(f"rank_score: {row['_rank_score']}\n")
            f.write(f"original_image: {copied_orig.name}\n")
            f.write(f"overlay_image:  {overlay_path.name}\n")
            f.write(f"gaps_image:     {gaps_img_path.name}\n\n")

            # CSVの主要情報
            def w(key):
                if key in row and pd.notna(row[key]):
                    f.write(f"{key}: {row[key]}\n")
            f.write("## CSV row fields\n")
            for key in ["path","W","H", gx,gy,gr, px,py,pr,
                        "center_err_px","radius_err_px","center_err_rel_r","radius_err_rel_r","iou","success"]:
                if key and key in df.columns:
                    w(key)
            f.write("\n")

            # ランドルト環（外側→内側）
            f.write("## Landolt rings (outer → inner)\n")
            # radii は result["gaps"] のキー（ギャップがあった半径）のみ記録されている
            # 「外側から抽出されたランドルト環の半径，空き方向(0..11時)」
            radii = sorted(result["gaps"].keys(), reverse=True)
            if not radii:
                f.write("(ギャップ検出なし)\n")
            else:
                for r in radii:
                    hours = result["gap_hours"].get(r, [])
                    hours_str = ",".join(str(h) for h in hours) if hours else "-"
                    f.write(f"r={r:.1f}px : empty_hours={hours_str}\n")

        print(f"[OK] {out_dir} に保存完了")

    print("すべて完了。")
def analyze_and_draw(image_path, pr_x, pr_y, pr_r,
                     ring_is_bright="auto",
                     n_samples=720,
                     smooth_deg=6.0, k_std=0.6, min_gap_deg=8.0,
                     band_frac=0.05,
                     band_radial_samples=9,
                     bg_probe_frac=0.01,
                     bg_gap_px=2.0,
                     max_rings=5,
                     scan_inner_min_frac=0.03,
                     n_scan=48,
                     min_rad_sep_frac=0.45,
                     out_path="__gaps_debug.png", try_show=False,
                     scale=2.0,
                     # 暫定: “2時間大きめ”を打ち消す（安定後は None）
                     angle_offset_hours_override=-2,
                     offset_outer_weight=2.0):
    import math

    # --- helpers ---
    def ring_band_px(r, short_side, frac_max=0.05, frac_min=0.015, rel=0.12):
        # 小さな円では細め、大きい円では太め
        return float(np.clip(max(short_side*frac_min, rel*float(r)),
                             a_min=2.0, a_max=short_side*frac_max))

    def refine_gap_center(theta_deg, vals, st_deg, ed_deg, ring_is_bright):
        n = len(vals)
        ang = np.asarray(theta_deg)
        if ed_deg >= st_deg:
            mask = (ang >= st_deg) & (ang < ed_deg)
        else:
            mask = (ang >= st_deg) | (ang < ed_deg)
        idxs = np.where(mask)[0]
        if idxs.size < 3:
            return (st_deg + ((ed_deg - st_deg) % 360.0)/2.0) % 360.0
        sel = vals[idxs]
        k_in = idxs[np.argmin(sel)] if ring_is_bright else idxs[np.argmax(sel)]
        km1 = (k_in - 1) % n; kp1 = (k_in + 1) % n
        y1, y2, y3 = float(vals[km1]), float(vals[k_in]), float(vals[kp1])
        denom = (y1 - 2.0*y2 + y3)
        frac = 0.0 if abs(denom) < 1e-9 else 0.5*(y1 - y3)/denom
        frac = float(np.clip(frac, -0.5, 0.5))
        k_star = (k_in + frac) % n
        return (k_star / n) * 360.0

    def angle_to_clock_hour_with_offset(angle_deg, offset_deg=0.0, return_float=False):
        h = ((angle_deg + 90.0 + offset_deg) % 360.0) / 30.0
        if return_float:
            return h % 12.0
        return int((h + 0.5)) % 12

    def gap_mask(n, st, ed):
        a = (np.arange(n) / n) * 360.0
        st = st % 360.0; ed = ed % 360.0
        if (ed - st) % 360.0 == 0:
            return np.ones(n, dtype=bool)
        if ed >= st:
            return (a >= st) & (a < ed)
        else:
            return (a >= st) | (a < ed)

    def complement_intervals(gaps):
        if not gaps:
            return [(0.0, 360.0)]
        gaps = [(st % 360.0, ed % 360.0) for st, ed in gaps]
        pts = []
        for st, ed in gaps:
            pts.append((st % 360.0, 1))
            pts.append((ed % 360.0, -1))
        pts.sort()
        merged = []
        acc = 0; cur_start = None
        for ang, delta in pts + [(pts[0][0] + 360.0, 0)]:
            prev_acc = acc
            acc += delta
            if prev_acc == 0 and acc > 0:
                cur_start = ang
            elif prev_acc > 0 and acc == 0 and cur_start is not None:
                merged.append((cur_start, ang))
                cur_start = None
        ok = []
        if not merged:
            ok = [(0.0, 360.0)]
        else:
            cur = 0.0
            for st, ed in merged:
                if (st - cur) % 360.0 > 1e-6:
                    ok.append((cur % 360.0, st % 360.0))
                cur = ed
            if (360.0 - cur) % 360.0 > 1e-6:
                ok.append((cur % 360.0, 360.0))
        return ok

    def draw_arc(ax, cx, cy, r, st, ed, color, lw, z=5):
        ang0 = -ed
        width = (ed - st) % 360.0
        ax.add_patch(Arc((cx, cy), 2*r, 2*r, angle=0,
                         theta1=ang0, theta2=ang0 - width,
                         color=color, lw=lw, zorder=z))

    def polar_to_xy(cx, cy, r, deg):
        th = math.radians(deg)
        return cx + r * math.cos(th), cy + r * math.sin(th)

    def _screen_wh(default=(1920, 1080)):
        try:
            import tkinter as tk
            r = tk.Tk(); r.withdraw()
            w, h = r.winfo_screenwidth(), r.winfo_screenheight()
            r.destroy()
            return int(w), int(h)
        except Exception:
            return default

    # --- load & prep ---
    img = mpimg.imread(image_path)
    means = image_means(img)
    gray = means["gray"].astype(np.float32)
    H, W = gray.shape
    short_side = float(min(H, W))

    # 代表帯厚（外周ベース）でスキャン範囲を決める
    band_outer_px = ring_band_px(pr_r, short_side)
    r_center_max = max(0.0, pr_r - band_outer_px/2.0 - 1.0)
    r_center_min = max(band_outer_px/2.0 + 1.0, pr_r * scan_inner_min_frac)
    if r_center_max <= r_center_min + 1.0:
        r_center_min = max(1.0 + band_outer_px/2.0, min(r_center_max, pr_r*0.6))

    # figure
    scr_w, scr_h = _screen_wh()
    margin = 0.9
    max_w_px = scr_w * margin; max_h_px = scr_h * margin
    scale_cap = min(max_w_px / max(W, 1), max_h_px / max(H, 1))
    actual_scale = max(min(float(scale), float(scale_cap)), 0.5)
    dpi = plt.rcParams.get("figure.dpi", 100)
    fig_w, fig_h = (W / dpi * actual_scale, H / dpi * actual_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img); ax.set_title(os.path.basename(image_path))
    ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(Circle((pr_x, pr_y), pr_r, fill=False, linewidth=2, edgecolor="b", zorder=2))
    ax.plot([pr_x], [pr_y], "o", markersize=4, color="b", zorder=3)

    # 背景プローブ → 白/黒自動判定
    if ring_is_bright == "auto":
        probe_px = max(2.0, short_side * float(bg_probe_frac))
        r_in_probe  = pr_r + float(bg_gap_px)
        r_out_probe = pr_r + float(bg_gap_px) + probe_px
        bg_mean = annulus_mean_gray(gray, pr_x, pr_y, r_in_probe, r_out_probe,
                                    n_theta=max(360, n_samples//2), n_r=5)
        img_mean = float(gray.mean())
        if bg_mean <= 0.45:
            ring_is_bright_bool = True
        elif bg_mean >= 0.55:
            ring_is_bright_bool = False
        else:
            ring_is_bright_bool = (bg_mean < img_mean)
    else:
        ring_is_bright_bool = bool(ring_is_bright)

    # --- scan radii (inner only) ---
    r_candidates = np.linspace(r_center_max, r_center_min, max(2, int(n_scan)))
    cand = []
    for rc in r_candidates:
        # 必ず外側の円の内側で完結
        if rc + 1.0 >= pr_r:  # ほんの余白
            continue
        band_px = ring_band_px(rc, short_side)
        if rc - band_px/2.0 <= 0.5 or rc + band_px/2.0 >= pr_r - 0.5:
            continue

        theta_deg, vals = band_intensity_profile(
            gray, pr_x, pr_y, rc, band_px=band_px,
            n_radial=band_radial_samples, n_samples=n_samples
        )
        gaps, _stats = detect_gaps_on_ring(
            vals,
            ring_is_bright=ring_is_bright_bool,
            smooth_deg=smooth_deg, k_std=k_std, min_gap_deg=min_gap_deg
        )
        if not gaps:
            continue

        def _gap_width(seg): return (seg[1] - seg[0]) % 360.0
        gmax = max(gaps, key=_gap_width)
        width_deg = _gap_width(gmax)

        m = gap_mask(len(vals), gmax[0], gmax[1])
        if m.any() and (~m).any():
            contrast = abs(float(vals[~m].mean()) - float(vals[m].mean()))
        else:
            contrast = float(vals.std())

        mid_ref = refine_gap_center(theta_deg, vals, gmax[0], gmax[1], ring_is_bright_bool)
        score = float(width_deg * (contrast + 1e-6))

        cand.append(dict(
            r=float(rc),
            gaps=[gmax],
            width_deg=width_deg,
            contrast=contrast,
            score=score,
            theta=theta_deg,
            vals=vals,
            band_px=band_px
        ))

    # --- NMS on radius, then keep top-K ---
    cand.sort(key=lambda d: d["score"], reverse=True)
    def min_sep_for_pair(r1, r2):
        return max(2.0, min(ring_band_px(r1, short_side), ring_band_px(r2, short_side)) * float(min_rad_sep_frac))

    keep = []
    for c in cand:
        if len(keep) >= int(max_rings):
            break
        ok = True
        for k in keep:
            if abs(c["r"] - k["r"]) < min_sep_for_pair(c["r"], k["r"]):
                ok = False; break
        if ok:
            keep.append(c)

    # --- auto/override angle offset ---
    mid_list, weight_list = [], []
    for c in keep:
        st, ed = c["gaps"][0]
        rc     = float(c["r"])
        th     = c["theta"]; vs = c["vals"]
        mid_ref = refine_gap_center(th, vs, st, ed, ring_is_bright_bool)
        w = max(1e-6, c["width_deg"] * c["contrast"] * (1.0 + offset_outer_weight*(rc/pr_r)))
        mid_list.append(mid_ref); weight_list.append(w)

    if angle_offset_hours_override is None and weight_list:
        def nearest_int(x): return int((x + 0.5)) % 12
        diffs = []
        for mid, w in zip(mid_list, weight_list):
            h_float = angle_to_clock_hour_with_offset(mid, offset_deg=0.0, return_float=True)
            d = nearest_int(h_float) - h_float
            if d > 6:  d -= 12
            if d < -6: d += 12
            diffs.append((d, w))
        delta_h = sum(d*w for d, w in diffs) / max(1e-6, sum(w for _, w in diffs))
        offset_deg = float(delta_h) * 30.0
    else:
        offset_deg = 30.0 * float(angle_offset_hours_override or 0.0)

    print(f"[INFO] angle offset used = {offset_deg:.2f} deg")

    # --- draw ---
    def draw_compass(ax, cx, cy, r, offset_deg):
        for k in range(12):
            ang = (k * 30.0 - 90.0 - offset_deg) % 360.0
            th  = math.radians(ang)
            r0  = r * 1.10
            r1  = r * (1.16 if k % 3 == 0 else 1.13)
            x0, y0 = cx + r0 * math.cos(th), cy + r0 * math.sin(th)
            x1, y1 = cx + r1 * math.cos(th), cy + r1 * math.sin(th)
            ax.plot([x0, x1], [y0, y1], "-", lw=(2 if k % 3 == 0 else 1),
                    color="white", alpha=0.7, zorder=9)
            if k % 3 == 0:
                hx, hy = cx + (r1*1.08) * math.cos(th), cy + (r1*1.08) * math.sin(th)
                label = {0:"12",3:"3",6:"6",9:"9"}[k]
                ax.text(hx, hy, label, color="white", fontsize=10, ha="center", va="center",
                        zorder=10, bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.35, linewidth=0))

    if keep:
        draw_compass(ax, pr_x, pr_y, keep[0]["r"], offset_deg)

    keep.sort(key=lambda d: d["r"], reverse=True)
    for c in keep:
        rc = c["r"]
        # 非欠け=シアン
        for st_ok, ed_ok in complement_intervals(c["gaps"]):
            if (ed_ok - st_ok) % 360.0 < 1e-3: continue
            draw_arc(ax, pr_x, pr_y, rc, st_ok, ed_ok, color=(0.2, 1.0, 1.0), lw=1.5, z=4)
        # 欠け=赤 + ラベル
        for st, ed in c["gaps"]:
            draw_arc(ax, pr_x, pr_y, rc, st, ed, color="r", lw=3.0, z=6)
            for ang in (st, ed):
                x, y = polar_to_xy(pr_x, pr_y, rc, ang)
                ax.plot([x], [y], "o", color="r", markersize=3.5, zorder=7)
            mid_ref = refine_gap_center(c["theta"], c["vals"], st, ed, ring_is_bright_bool)
            hr = angle_to_clock_hour_with_offset(mid_ref, offset_deg=offset_deg, return_float=False)
            tx, ty = polar_to_xy(pr_x, pr_y, rc*1.06, mid_ref)
            ax.text(tx, ty, f"{hr}", color="r", fontsize=9,
                    ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, linewidth=0))

    shown = False
    if try_show:
        try:
            plt.show(block=True); shown = True
        except Exception as e:
            print(f"GUI表示できなかったため保存します: {out_path}\n理由: {e}")
    if not shown:
        fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0)
        print(f"結果画像を保存しました: {out_path}")
    plt.close(fig)

    # --- return ---
    gaps_map = {c["r"]: c["gaps"] for c in keep}
    hours_map = {}
    used_band_px = {}
    for c in keep:
        st, ed = c["gaps"][0]
        mid_ref = refine_gap_center(c["theta"], c["vals"], st, ed, ring_is_bright_bool)
        hr = angle_to_clock_hour_with_offset(mid_ref, offset_deg=offset_deg, return_float=False)
        hours_map[c["r"]] = [hr]
        used_band_px[c["r"]] = c.get("band_px", ring_band_px(c["r"], short_side))

    return {
        "mean_colors": {k: v for k, v in means.items() if k != "gray"},
        "gaps": gaps_map,
        "gap_hours": hours_map,
        "radii_outer_to_inner": [c["r"] for c in keep],
        "used_band_px": used_band_px,      # 半径ごとの帯厚（辞書）
        "ring_is_bright": ring_is_bright_bool,
        "bg_mean_outside": (bg_mean if ring_is_bright == "auto" else None),
        "angle_offset_deg": offset_deg,
    }


if __name__ == "__main__":
    main()
