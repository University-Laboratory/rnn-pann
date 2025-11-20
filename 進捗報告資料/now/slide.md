# 進捗報告 ?/?(?)

長崎大学工学部工学科 B4 丸田研究室\
35221011 大塚直哉

---

# GRU + PANN

## やったこと

### 結果

---

# 回路パラメータの初期値、学習率の決定方法

## ざっくり解析的に各パラメータを決める

### オイラー法の式

$$i_L(t + \Delta t) = i_L(t) + \frac{\Delta t}{L} (V_{in} * u(t) - v_c)$$

$$v_C(t + \Delta t) = v_C(t) + \frac{\Delta t}{C} \left( i_L - \frac{v_C}{R} \right)$$

### R

定常状態では、以下の近似が成り立つと仮定

$$
i_L(t+\Delta t) \approx i_L(t)
$$

$$
v_C(t+\Delta t) \approx v_C(t)
$$

オイラー法の式に代入

> $$v_C(t + \Delta t) = v_C(t) + \frac{\Delta t}{C} \left( i_L - \frac{v_C}{R} \right)$$

$$
0 = \frac{\Delta t}{C}\left(i_L - \frac{v_C}{R}\right)
$$

$$
R \approx \frac{v_C}{i_L}
$$

### L

> $$i_L(t+\Delta t) = i_L(t) + \frac{\Delta t}{L} (V_{in}u(t)-v_C(t))$$

$$
\Delta i_L = i_L(t+\Delta t)-i_L(t)
$$

$$
x = \Delta t (V_{in}u(t)-v_C(t))
$$

とおいたとき、

$$
\Delta i_L = \frac{1}{L} \cdot x
$$

となるので、これを最小二乗法で解くと、

$$
\frac{1}{L} = \frac{\sum \Delta i_L \cdot x}{\sum x^2}
$$

### C

> $$v_C(t+\Delta t) = v_C(t) + \frac{\Delta t}{C}\left(i_L(t)-\frac{v_C(t)}{R}\right)$$

$$
\Delta v_C = v_C(t+\Delta t) - v_C(t)
$$

$$
z = \Delta t\left(i_L - \frac{v_C}{R_0}\right)
$$

とおいたとき、これを最小二乗法で解くと、

$$
\Delta v_C = \frac{1}{C} z
$$

$$
\frac{1}{C} = \frac{\sum \Delta v_C \cdot z}{\sum z^2}
$$

### 結果

大体いい感じに求まる

# NA

- GRU 早めにやる
- v_s, u_t は csv に多分書いてる
- t の間隔違うのはいい感じにモデルに組み込んでみる
- 一周期のプロット数減らす
- 不連続モードと連続のやつ両方で学習しないとだめ
- 定常で学習して過渡をシミュレーションできるのか？
- ホイン法でモデリングする
- プッシュプルコンバータもやる
