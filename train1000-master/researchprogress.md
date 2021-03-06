
<!-- footer: 2019/06/13 -->
<!-- page_number: true -->
# 研究調査
##### 2019/06/13
## ここまでの顛末
1000個のtrainデータで学習をする[train with 1000](http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/)のサンプルコードを元に、モデルのカスタマイズを試していた。

自分で作成したcifar-10用の簡単なモデルでー(test:0.9, val:0.4）


使用しているデータセットが[cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)なので、入力画像のピクセル数が畳み込みによって、既存モデルの場合、わずか中盤でピクセル数が(1,1)になってしまう。そこからさらに畳み込みを行う為、テストデータ、評価データでの誤差が双方落ちていると見た。

---


# 調べたこと
少ないデータでの学習はモデルよりもデータの質に左右されることが多いらしい。今現在、効果が確認できている手法は下記二つ
#### 1. Auto argumentation
データのコントラスト補正等の画像処理を行う。組み合わせは10の32乗個あるらしい。それを強化学習で最適な画像処理の組み合わせを見つける手法。
#### 2. mixup　（BC learning）
少ないデータを複製してコントラスト補正等、拡大縮小、回転した画像もデータセットとして扱う、擬似的にデータセットを増やす手法

連鎖的なデータ増強処理中のどのタイミングでBC-Learningのmixを行うかによって、最終的な分類精度が変化することをデータセットによって確認できたりできなかったりらしい。`要調査`

---

## 効果が確認できなかった手法
### 1. ARS-aug
転移学習を除いて現時点のCIFAR-10/100において最も高い分類精度を達成した手法だが、augumentは、`データの傾向が変われば、適したデータ増強手法も変わる`ので、データが1000枚に減ったことで傾向が変わった。2％くらいしか上がらなかったらしい。

---

## うまくいくかもしれない手法
### 1. Residual Dense Net-SD
転移学習や特別なデータ増強等の手法なしに唯一、表1の最高分類精度競争に名を刻む手法。半分程度のパラメータでそれを達成する。実装公開ナシのため未検証(2018/12/28時点)

### 2. ProxylessNAS
データセットに対して最適なモデル構造を探索する手法の1つで、強化学習/勾配計算で構造を探索する。Residual Dense Netよりもパラメータが少ないらしい

---

## 現在のビジョン
**データ数が少ない学習**に関しては、モデルよりデータの質らしい(未検証)。１つのデータセットにおいて、前処理もしっかりして計算しやすいようにして、特徴量が掴みやすいような画像処理をかけてやる。

とはいえ、モデルもおろそかにしてはいけない気もする。

**`Auto-argument, BC-learning等でデータを整え、ProxylessNASで最適なモデル構造を探索する`**


**`既存のモデルを重み付きで再利用し、1000個のデータのみでそれに最適なモデルに改造するファインチューニング`**

という２つのアプローチを考えついた。
２つめも考慮し、**データセットは32 * 32 のcifar-10ではなく、ピクセル数100~200の物を使うことにする**。

参考文献：  
https://qiita.com/imenurok/items/31490be74f3437dc8fed
