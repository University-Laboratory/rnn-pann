# 進捗報告 ?/?(?)

長崎大学工学部工学科 B4 丸田研究室\
35221011 大塚直哉

---

# フロー

- dt 固定
- 移動平均に修正
- 標準化、無次元化のあることによるメリット

```mermaid

flowchart TB

    subgraph 加工前
        A[CSV]
        t_raw[t_raw]
        dt_raw[dt_raw]
        iL_raw[iL_raw]
        vC_raw[vC_raw]
        u_raw[u_raw 作成]
        vs_raw[vs_raw 作成]
    end

    A --> t_raw
    t_raw --> dt_raw
    A --> iL_raw
    A --> vC_raw


    subgraph ダウンサンプリング
        dt_downsampled[dt_ds]
        iL_downsampled[iL_ds]
        vC_downsampled[vC_ds]
        u_downsampled[u_ds]
        vs_downsampled[vs_ds]
    end

    dt_raw --> dt_downsampled
    iL_raw --> iL_downsampled
    vC_raw --> vC_downsampled
    u_raw --> u_downsampled
    vs_raw --> vs_downsampled

    %% ここまで共通処理 %%

    subgraph BuckConverterCellの処理
        subgraph フィルタリング
            iL_filt[iL_filt]
            vC_filt[vC_filt]
        end
        iL_downsampled --> iL_filt
        vC_downsampled --> vC_filt

        subgraph n周期切り出し
            dt_buck[dt_buck]
            iL_buck[iL_buck]
            vC_buck[vC_buck]
            u_buck[u_buck]
            vs_buck[vs_buck]
        end
        dt_downsampled --> dt_buck
        iL_filt --> iL_buck
        vC_filt --> vC_buck
        u_downsampled --> u_buck
        vs_downsampled --> vs_buck

        BuckConverterCell[BuckConverterCellの学習]
        iL_buck --> BuckConverterCell
        vC_buck --> BuckConverterCell
        u_buck --> BuckConverterCell
        vs_buck --> BuckConverterCell
        dt_buck --> BuckConverterCell

        subgraph 推論値
            BuckConverterCell_out[iL_pred, vC_pred]
        end

        BuckConverterCell --> BuckConverterCell_out

        subgraph パラメータの推論
            BuckConverterCell_params[L_pred, C_pred, R_pred]
        end
        BuckConverterCell --> BuckConverterCell_params
    end

    subgraph GRUの処理
        subgraph ダウンサンプリングコピー
            dt_downsampled_copy[dt_ds]
            u_downsampled_copy[u_ds]
            vs_downsampled_copy[vs_ds]
            iL_downsampled_copy[iL_ds]
            vC_downsampled_copy[vC_ds]
        end
        dt_downsampled --> dt_downsampled_copy

        subgraph ノイズ抽出
            noise_data[iL_noise, vC_noise]
        end
        BuckConverterCell_out --> noise_data
        iL_downsampled_copy --> noise_data
        vC_downsampled_copy --> noise_data

        subgraph 無次元化
            iL_nd[iL_nd]
            vC_nd[vC_nd]
            vs_nd[vs_nd]
            dt_nd[dt_nd]
        end
        iL_downsampled_copy --> iL_nd
        vC_downsampled_copy --> vC_nd
        vs_downsampled_copy --> vs_nd
        dt_downsampled_copy --> dt_nd

        GRU[GRU学習]
        iL_nd --> GRU
        vC_nd --> GRU
        vs_nd --> GRU
        dt_nd --> GRU
        noise_data --> GRU

        GRU_out[GRUのノイズ予測]
        GRU --> GRU_out
    end


    FinalOutPut[BuckConverterCell + GRU の結果]
    GRU_out --> FinalOutPut
    BuckConverterCell_out --> FinalOutPut

```
