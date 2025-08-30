#######################################
#   Takako Yamada 20250830 version
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

print("Matplotlib backend:", mpl.get_backend())   # デバッグ表示

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Arc


# =====================
# 設定
# =====================
IN_CSV    = "test_metrics.csv"
OUT_CSV   = "test_metrics_filename.csv"  # path列をファイル名化した全件の保存先
IMG_DIR   = "images"                  # 画像探索ルート
TOP_K     = 10                         # 最小誤差で抽出する件数
OUT_ROOT  = "picked_top10"             # 出力ルートフォルダ
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
    max_rings=5,          # ★追加
    scan_inner_min_frac=0.15,  # 必要なら調整
    n_scan=28,
    min_rad_sep_frac=0.6,
    scale=2.0,
    binary_threshold_ratio=0.65,  # 二値化閾値（0.0-1.0、0.65=65%）
)


# =====================
# ユーティリティ
# =====================
def basename_series(path_series: pd.Series) -> pd.Series:
    """パス文字列列からファイル名だけを安全に取り出す（区切りを/に正規化）"""
    p = (
        path_series.astype(str)
        .str.strip()
        .str.lstrip("\ufeff")          # 先頭BOM対策
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
    rs      = np.linspace(r_inner, r_outer, max(1, int(n_r)))
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
            f.write(f"gaps_image:     {gaps_img_path.name}\n")
            # analyze_and_draw からの戻り値を利用
            f.write(f"binary_image:   {os.path.basename(result.get('binary_image_path', 'N/A'))}\n")
            f.write(f"profiles_image: {os.path.basename(result.get('profiles_image_path', 'N/A'))}\n")
            f.write(f"binary_threshold_ratio: {GAP_PARAMS.get('binary_threshold_ratio', 0.65):.3f}\n\n")

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
                     angle_offset_hours_override=None, # デフォルトをNoneに変更
                     offset_outer_weight=2.0,
                     binary_threshold_ratio=0.65):  # 二値化閾値（0.0-1.0）
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
        # gapsが空の場合は全周が非欠けとみなす
        if not gaps:
            return [(0.0, 360.0)]

        # ギャップの開始と終了点をフラット化し、種類（開始/終了）を付与
        pts = []
        for st, ed in gaps:
            pts.append((st % 360.0, 1))   # ギャップ開始
            pts.append((ed % 360.0, -1))  # ギャップ終了
        pts.sort()

        # ギャップではない領域（補集合）を抽出
        complements = []
        current_overlap_count = 0
        current_start_angle = 0.0

        if not pts: # gapsが空でなくとも、重複などでptsが空になりうる
            return [(0.0, 360.0)]

        # 0度から最初の点までの領域を考慮（ギャップでなければ）
        if pts[0][0] > 1e-6 and pts[0][1] == 1: # 最初のギャップが0度から始まらない
            complements.append((0.0, pts[0][0]))

        for i in range(len(pts)):
            angle, delta = pts[i]
            # 前の区間がギャップでなかった場合（current_overlap_count == 0）
            # かつ、新しい区間が始まる場合（current_start_angle < angle）
            if current_overlap_count == 0 and angle > current_start_angle + 1e-6:
                complements.append((current_start_angle, angle))

            current_overlap_count += delta
            current_start_angle = angle # 現在の点を次の開始点として更新

            # 360度をまたぐケースの考慮
            if i == len(pts) - 1 and current_overlap_count == 0 and complements:
                # 最後の点から360度までがギャップでなければ追加
                if complements[-1][1] < 360.0 - 1e-6:
                    complements.append((complements[-1][1], 360.0))
                
        # 360度をまたぐギャップの処理（例: 350度から10度）
        # もしギャップの始まりが終わりより大きい場合
        wrapped_complements = []
        for st, ed in complements:
            if ed < st: # 360度をまたぐ場合
                wrapped_complements.append((st, 360.0))
                wrapped_complements.append((0.0, ed))
            else:
                wrapped_complements.append((st, ed))

        # ソートしてマージ (微調整)
        if not wrapped_complements: return [(0.0, 360.0)]
        wrapped_complements.sort()
        
        final_complements = []
        if wrapped_complements:
            current_st, current_ed = wrapped_complements[0]
            for i in range(1, len(wrapped_complements)):
                st, ed = wrapped_complements[i]
                if st <= current_ed + 1e-6: # 重なるか連続している場合
                    current_ed = max(current_ed, ed)
                else: # 離れている場合
                    final_complements.append((current_st, current_ed))
                    current_st, current_ed = st, ed
            final_complements.append((current_st, current_ed))

        # 0.0 - 360.0 の範囲に正規化
        final_complements = [(s % 360.0, e % 360.0) for s, e in final_complements]
        return final_complements


    def draw_arc(ax, cx, cy, r, st, ed, color, lw, z=5):
        ang0 = -ed # matplotlibのArcは反時計回りで、theta1が開始
        width = (ed - st) % 360.0
        # 角度計算の調整: matplotlibは反時計回り、0度が右方向。画像は左方向が0度で時計回り。
        # stとedは画像座標系（左が0度で時計回り）なので、これをmatplotlib用に変換
        # 0度を右に、時計回りを反時計回りに変換するために、角度を負にする
        theta1 = 90 - ed # 画像のedがArcの開始
        theta2 = 90 - st # 画像のstがArcの終了
        
        # 360度をまたぐArcの描画は複雑になるため、単純に描画
        if ed < st: # 360度をまたぐ場合（例: st=300, ed=30）
            # 300度から360度まで
            ax.add_patch(Arc((cx, cy), 2*r, 2*r, angle=0,
                             theta1=90-360, theta2=90-st, # 90-st, 90-360 のArc
                             color=color, lw=lw, zorder=z))
            # 0度から30度まで
            ax.add_patch(Arc((cx, cy), 2*r, 2*r, angle=0,
                             theta1=90-ed, theta2=90-0, # 90-0, 90-ed のArc
                             color=color, lw=lw, zorder=z))
        else:
            ax.add_patch(Arc((cx, cy), 2*r, 2*r, angle=0,
                             theta1=theta1, theta2=theta2,
                             color=color, lw=lw, zorder=z))

    def polar_to_xy(cx, cy, r, deg):
        th = math.radians(deg)
        return cx + r * math.cos(th), cy + r * math.sin(th)

    def _screen_wh(default=(1920, 1080)):
        # Tkinterを使わないように変更（Aggバックエンド利用時不要のため）
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

    # --- 外周円から中心に向かって等間隔で円を生成して番号付け ---
    def generate_numbered_circles(cx, cy, outer_r, n_circles=20):
        """外周円から中心に向かって等間隔で円を生成して番号をつける"""
        # スキャン範囲を設定（外周円から中心に向かって）
        r_candidates = np.linspace(outer_r * 0.95, outer_r * 0.1, n_circles)
        
        print(f"[INFO] Generating {len(r_candidates)} numbered circles from {r_candidates[0]:.1f} to {r_candidates[-1]:.1f}")
        
        numbered_circles = []
        
        for i, r in enumerate(r_candidates):
            # 円周上の強度プロファイルを取得
            theta_deg, vals = ring_intensity_profile(gray, cx, cy, r, n_samples=360)
            
            # 二値化の閾値を自動決定
            vals_sorted = np.sort(vals)
            threshold_idx = int(len(vals_sorted) * 0.4)  # 40%の位置を閾値に
            threshold = vals_sorted[threshold_idx]
            
            # 黒と白のピクセルを分類
            black_pixels = vals < threshold
            white_pixels = vals >= threshold
            
            # 黒の割合を計算
            black_ratio = np.sum(black_pixels) / len(vals)
            white_ratio = 1.0 - black_ratio
            
            # 円の情報を記録
            circle_info = {
                'number': i + 1,  # 1から始まる番号
                'r': r,
                'black_ratio': black_ratio,
                'white_ratio': white_ratio,
                'theta_deg': theta_deg,
                'vals': vals,
                'threshold': threshold,
                'score': black_ratio  # スコア（黒の割合）
            }
            
            numbered_circles.append(circle_info)
            
            print(f"[INFO] Circle #{i+1}: r={r:.1f}, black_ratio={black_ratio:.3f}, white_ratio={white_ratio:.3f}")
        
        return numbered_circles
    
    # 番号付き円を生成
    numbered_circles = generate_numbered_circles(
        pr_x, pr_y, pr_r, 
        n_circles=20  # 20個の円を生成
    )
    
    print(f"[INFO] Generated {len(numbered_circles)} numbered circles")
    
    # 外周円の内側だけで二値化を適用
    def binarize_inside_circle(gray_image, cx, cy, outer_r, threshold_ratio):
        """外周円の内側だけで二値化を適用"""
        H, W = gray_image.shape
        
        # 外周円の内側のマスクを作成
        y_coords, x_coords = np.ogrid[:H, :W]
        distance_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        inside_mask = distance_from_center <= outer_r
        
        # 外周円の内側のピクセル値のみを取得
        inside_pixels = gray_image[inside_mask]
        
        if len(inside_pixels) == 0:
            print("[WARNING] No pixels inside the outer circle")
            return np.zeros_like(gray_image, dtype=bool), 0.0
        
        # 内側のピクセル値で閾値を決定
        vals_sorted = np.sort(inside_pixels)
        threshold_idx = int(len(vals_sorted) * threshold_ratio)
        threshold = vals_sorted[threshold_idx]
        
        # 外周円の内側のみ二値化
        binary_image = np.zeros_like(gray_image, dtype=bool)
        binary_image[inside_mask] = gray_image[inside_mask] < threshold
        
        return binary_image, threshold
    
    # 外周円の内側だけで二値化
    binary_gray, global_threshold = binarize_inside_circle(gray, pr_x, pr_y, pr_r, binary_threshold_ratio)
    print(f"[INFO] Inside circle binarization ({binary_threshold_ratio*100:.0f}% threshold): {global_threshold:.3f}")
    
    # 二値化画像を保存
    binary_save_path = out_path.replace(".png", "_binary.png")
    def save_binary_image(binary_img, save_path, cx, cy, outer_r, keep_circles):
        """二値化画像を保存（外周円の内側のみ）"""
        # 二値化画像を0-255の範囲に変換（黒=0, 白=255）
        binary_uint8 = (~binary_img).astype(np.uint8) * 255  # 反転して白=255にする
        
        # matplotlibで保存
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(binary_uint8, cmap='gray')
        ax.set_title(f"Binary Image Inside Circle ({binary_threshold_ratio*100:.0f}% threshold={global_threshold:.3f})")
        ax.axis("off")
        ax.set_aspect("equal")
        
        # 外周円を描画
        ax.add_patch(Circle((cx, cy), outer_r, fill=False, linewidth=2, edgecolor="b", zorder=2))
        ax.plot([cx], [cy], "o", markersize=4, color="b", zorder=3)
        
        # 選択されたランドルト環を描画
        for circle in keep_circles:
            rc = circle['r']
            circle_number = circle.get("circle_number", 0)
            ax.add_patch(Circle((cx, cy), rc, fill=False, linewidth=2, edgecolor="red", zorder=4))
            
            # 番号を表示
            tx, ty = polar_to_xy(cx, cy, rc*1.1, 90.0)
            ax.text(tx, ty, f"#{circle_number}", color="red", fontsize=12,
                            ha="center", va="center", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, linewidth=0))
        
        fig.savefig(save_path, dpi=120, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"[INFO] Binary image saved: {save_path}")
    
    # 結果をcand形式に変換（穴の情報は空にする）
    cand = []
    for circle in numbered_circles:
        cand.append(dict(
            r=float(circle['r']),
            gaps=[],  # 穴の情報は空
            width_deg=0.0,  # 穴の幅は0
            contrast=float(circle['black_ratio']),
            score=float(circle['score']),
            theta=circle['theta_deg'],
            vals=circle['vals'],
            band_px=ring_band_px(circle['r'], short_side),
            black_ratio=circle['black_ratio'],
            white_ratio=circle['white_ratio'],
            circle_number=circle['number'],  # 円の番号を追加
            binary_gray=binary_gray,  # 二値化画像を追加
            global_threshold=global_threshold  # グローバル閾値を追加
        ))

    # --- 結果の選択（11本目と18本目だけ） ---
    # 番号順にソート（外側から内側へ）
    cand.sort(key=lambda d: d["circle_number"])
    
    # 11本目と18本目だけを選択
    target_numbers = [11, 18]  # 11本目と18本目
    keep = [c for c in cand if c["circle_number"] in target_numbers]
    
    print(f"[INFO] Selected {len(keep)} circles: #11 and #18")
    
    # 二値化画像を保存（11本目と18本目が選択された後）
    save_binary_image(binary_gray, binary_save_path, pr_x, pr_y, pr_r, keep)
    
    # 各円の二値化プロファイルを保存
    profiles_save_path = out_path.replace(".png", "_profiles.png")
    def save_circle_profiles(circles, binary_img, save_path, cx, cy):
        """各円の二値化プロファイルを保存（外周円の内側の二値化）"""
        if not circles:
            return
            
        n_circles = len(circles)
        fig, axes = plt.subplots(n_circles, 1, figsize=(12, 3*n_circles))
        if n_circles == 1:
            axes = [axes]
        
        for i, circle in enumerate(circles):
            ax = axes[i]
            r = circle['r']
            circle_number = circle['circle_number']
            
            # 円周上の二値化値を取得
            theta_deg, vals = ring_intensity_profile(binary_img.astype(np.float32), cx, cy, r, n_samples=360)
            
            # プロファイルをプロット
            ax.plot(theta_deg, vals, 'b-', linewidth=1, label=f'Binary Profile (Inside Circle)')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({binary_threshold_ratio*100:.0f}%)')
            
            # 白い領域（穴）をハイライト
            white_regions = vals < 0.5
            for j in range(len(theta_deg)):
                if white_regions[j]:
                    ax.axvspan(theta_deg[j]-1, theta_deg[j]+1, alpha=0.3, color='yellow')
            
            ax.set_title(f'Circle #{circle_number} (r={r:.1f}) - Binary Profile Inside Circle')
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Binary Value')
            ax.set_xlim(0, 360)
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"[INFO] Circle profiles saved: {save_path}")
    
    # 円のプロファイルを保存
    save_circle_profiles(keep, binary_gray, profiles_save_path, pr_x, pr_y)
    
    # 11本目と18本目の円周上で白いところを穴として検出
    def detect_gaps_on_circle(circle_data, binary_image, min_gap_deg=10.0, max_gap_deg=90.0):
        """円周上で白いところを穴として検出（二値化画像の白い部分を参照）"""
        r = circle_data['r']
        cx, cy = pr_x, pr_y
        
        # 円周上の二値化値を取得
        theta_deg, vals = ring_intensity_profile(binary_image.astype(np.float32), cx, cy, r, n_samples=360)
        
        # 白い領域（穴）を検出（二値化画像では白がFalse、黒がTrue）
        # 二値化画像の値が0.5未満の部分を白い領域として検出
        white_pixels = vals < 0.5  # 白い領域を検出
        
        gaps = []
        n = len(white_pixels)
        
        # 円周を一周して白の連続領域を検出
        i = 0
        while i < n:
            if white_pixels[i]:  # 白の開始
                start_idx = i
                # 白の連続を検出
                while i < n and white_pixels[i]:
                    i += 1
                end_idx = i
                
                # 角度に変換
                start_deg = (start_idx / n) * 360.0
                end_deg = (end_idx / n) * 360.0
                gap_width = (end_deg - start_deg) % 360.0
                
                # 適切なサイズの穴のみを選択
                if min_gap_deg <= gap_width <= max_gap_deg:
                    gaps.append((start_deg, end_deg))
            
            i += 1
        
        # 360度をまたぐギャップの処理（例：350度から10度）
        if gaps:
            # 最初のギャップと最後のギャップが360度をまたいで結合される可能性をチェック
            first_gap = gaps[0]
            last_gap = gaps[-1]
            # もし最後のギャップの終了が360度近く、最初のギャップの開始が0度近くの場合
            if (last_gap[1] > 360 - min_gap_deg) and (first_gap[0] < min_gap_deg):
                # 結合された新しいギャップ
                combined_start = last_gap[0]
                combined_end = first_gap[1]
                combined_width = (combined_end + 360.0 - combined_start) % 360.0
                if min_gap_deg <= combined_width <= max_gap_deg:
                    gaps.pop(0) # 最初のギャップを削除
                    gaps.pop(-1) # 最後のギャップを削除
                    gaps.append((combined_start, combined_end)) # 結合したギャップを追加
                    gaps.sort() # ソート
        return gaps
    
    # 各円で穴を検出
    for circle in keep:
        gaps = detect_gaps_on_circle(circle, binary_gray, min_gap_deg=10.0, max_gap_deg=90.0)
        circle['gaps'] = gaps
        
        if gaps:
            # 最大の穴を選択
            def _gap_width(seg): return (seg[1] - seg[0]) % 360.0
            gmax = max(gaps, key=_gap_width)
            width_deg = _gap_width(gmax)
            circle['width_deg'] = width_deg
            
            # 穴の中心角度を計算
            mid_deg = (gmax[0] + gmax[1]) / 2.0 % 360.0
            circle['gap_center'] = mid_deg
            
            print(f"[INFO] Circle #{circle['circle_number']}: r={circle['r']:.1f}, black_ratio={circle.get('black_ratio', 0):.3f}, white_ratio={circle.get('white_ratio', 0):.3f}")
            print(f"[INFO]   Found {len(gaps)} gaps, largest gap: {width_deg:.1f}° at {mid_deg:.1f}°")
        else:
            circle['width_deg'] = 0.0
            circle['gap_center'] = None
            print(f"[INFO] Circle #{circle['circle_number']}: r={circle['r']:.1f}, black_ratio={circle.get('black_ratio', 0):.3f}, white_ratio={circle.get('white_ratio', 0):.3f}")
            print(f"[INFO]   No gaps found")

    # --- 角度オフセットの計算 ---
    if angle_offset_hours_override is None and keep:
        # 検出された穴の中心角度から時計の方向を推定
        gap_centers = []
        weights = []
        for c in keep:
            if c.get('gaps') and len(c['gaps']) > 0:
                # 最初のギャップを基準にする
                st, ed = c['gaps'][0] 
                mid_deg = (st + ed) / 2.0 % 360.0
                weight = c.get('width_deg', 0)  # 穴の幅を重みとして使用
                gap_centers.append(mid_deg)
                weights.append(weight)
        
        # 最も重みの大きい穴の方向を基準とする
        if weights:
            max_weight_idx = np.argmax(weights)
            reference_angle = gap_centers[max_weight_idx]
            # 12時の方向（90度）との差を計算
            offset_deg = (90.0 - reference_angle) % 360.0
            if offset_deg > 180.0:
                offset_deg -= 360.0
        else:
            offset_deg = 0.0
    else:
        # angle_offset_hours_override が指定されている場合、それを優先
        offset_deg = 30.0 * float(angle_offset_hours_override or 0.0)

    print(f"[INFO] angle offset used = {offset_deg:.2f} deg")

    # --- draw ---
    def draw_compass(ax, cx, cy, r, offset_deg):
        # ランドルト環の半径を基準にコンパスを描画
        compass_r = r * 1.2 # 環の少し外側に表示
        for k in range(12):
            ang = (k * 30.0 - 90.0 - offset_deg) % 360.0
            th  = math.radians(ang)
            r0  = compass_r * 1.0
            r1  = compass_r * (1.06 if k % 3 == 0 else 1.03)
            x0, y0 = cx + r0 * math.cos(th), cy + r0 * math.sin(th)
            x1, y1 = cx + r1 * math.cos(th), cy + r1 * math.sin(th)
            ax.plot([x0, x1], [y0, y1], "-", lw=(2 if k % 3 == 0 else 1),
                     color="white", alpha=0.7, zorder=9)
            if k % 3 == 0:
                hx, hy = cx + (compass_r*1.08) * math.cos(th), cy + (compass_r*1.08) * math.sin(th)
                label = {0:"12",3:"3",6:"6",9:"9"}[k]
                ax.text(hx, hy, label, color="white", fontsize=10, ha="center", va="center",
                         zorder=10, bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.35, linewidth=0))

    if keep: # ランドルト環が検出された場合のみコンパスを描画
        # 最も外側のランドルト環の半径をコンパスの基準にする
        outermost_ring_r = sorted([c["r"] for c in keep], reverse=True)[0]
        draw_compass(ax, pr_x, pr_y, outermost_ring_r, offset_deg)

    # 番号付き円を描画（穴の情報も表示）
    keep.sort(key=lambda d: d["circle_number"])  # 番号順にソート
    for c in keep:
        rc = c["r"]
        circle_number = c.get("circle_number", 0)
        black_ratio = c.get("black_ratio", 0)
        gaps = c.get("gaps", [])
        
        # 円を描画（黒の割合に応じて色を変える）
        if black_ratio > 0.6:
            circle_color = "red"  # 黒の割合が高い場合は赤
        elif black_ratio > 0.4:
            circle_color = "orange"  # 中程度はオレンジ
        else:
            circle_color = "cyan"  # 低い場合はシアン (非欠け=シアンをここに入れる)
        
        # 円全体（非欠け部分）を描画 (シアン)
        # ギャップがない場合、またはギャップが特定できない場合、円全体をシアンで描画
        if not gaps:
            ax.add_patch(Circle((pr_x, pr_y), rc, fill=False, linewidth=1.5, edgecolor=circle_color, zorder=4))
        else:
            # ギャップの補集合（非欠け部分）を描画
            for st_ok, ed_ok in complement_intervals(gaps):
                if (ed_ok - st_ok) % 360.0 < 1e-3: continue
                draw_arc(ax, pr_x, pr_y, rc, st_ok, ed_ok, color=circle_color, lw=1.5, z=4)

        # 番号を表示（12時の位置）
        tx, ty = polar_to_xy(pr_x, pr_y, rc*1.1, 90.0)  # 12時の位置
        ax.text(tx, ty, f"#{circle_number}", color="white", fontsize=8,
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor=circle_color, alpha=0.8, linewidth=0))
        
        # 黒の割合を表示（6時の位置）
        tx2, ty2 = polar_to_xy(pr_x, pr_y, rc*1.1, 270.0)  # 6時の位置
        ax.text(tx2, ty2, f"{black_ratio:.2f}", color="white", fontsize=6,
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.05", facecolor=circle_color, alpha=0.6, linewidth=0))
        
        # 穴を描画（黄色でマーキング + ラベル）
        if gaps:
            for st, ed in gaps:
                draw_arc(ax, pr_x, pr_y, rc, st, ed, color="yellow", lw=4.0, z=6)
            
                # 穴の中心角度を計算
                mid_deg = (st + ed) / 2.0 % 360.0
                hr = angle_to_clock_hour_with_offset(mid_deg, offset_deg=offset_deg, return_float=False)
                tx, ty = polar_to_xy(pr_x, pr_y, rc*1.06, mid_deg)
                ax.text(tx, ty, f"{hr}", color="black", fontsize=10, fontweight='bold',
                         ha="center", va="center", zorder=8,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8, linewidth=1, edgecolor="black"))
        
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
    circle_numbers = {}
    
    for c in keep:
        circle_numbers[c["r"]] = c.get("circle_number", 0)
        used_band_px[c["r"]] = c.get("band_px", ring_band_px(c["r"], short_side))
        
        # 穴の時間情報を計算
        if c.get("gaps") and len(c["gaps"]) > 0:
            # 最初のギャップを対象にする
            st, ed = c["gaps"][0] 
            mid_deg = (st + ed) / 2.0 % 360.0
            hr = angle_to_clock_hour_with_offset(mid_deg, offset_deg=offset_deg, return_float=False)
            hours_map[c["r"]] = [hr] # ここもリストにする
        else:
             hours_map[c["r"]] = [] # ギャップがない場合は空リスト

    return {
        "mean_colors": {k: v for k, v in means.items() if k != "gray"},
        "gaps": gaps_map,
        "gap_hours": hours_map,
        "radii_outer_to_inner": [c["r"] for c in keep],
        "used_band_px": used_band_px,      # 半径ごとの帯厚（辞書）
        "ring_is_bright": ring_is_bright_bool,
        "bg_mean_outside": (bg_mean if ring_is_bright == "auto" else None),
        "angle_offset_deg": offset_deg,
        "binary_image_path": binary_save_path, # 追加
        "profiles_image_path": profiles_save_path, # 追加
        "black_white_ratios": {c["r"]: {"black": c.get("black_ratio", 0), "white": c.get("white_ratio", 0)} for c in keep},
        "circle_numbers": circle_numbers,  # 円の番号
    }


if __name__ == "__main__":
    main()