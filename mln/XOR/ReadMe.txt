・最急降下法を用いた Multi Layer Network を実装
・XORの学習／テストを実装
・学習は ./learn.py で実行
	＊入力層、中間層、出力層の３層で、ニューロン素子はそれぞれ２、３、１個で、全結合
		＋ニューラルネットとしては２層
	＊bios は全ての層で一定値、学習対象外
	＊学習率：0.15
	＊活性化関数は、中間層も出力層も tanh
	＊訓練データ：input_data = [[-1., -1.], [-1., 1.], [ 1., -1.], [ 1., 1.]]
		＋上記４つの訓練データを順番に全て50,000回提示
	＊教師データ：teach_data = [ [-1.], [ 1.], [ 1.], [-1.]]
	＊ネットワークの状態をダンプファイルで出力
		＋default-xor.dump：初期化直後のネットワークの状態をダンプしたファイル
		＋learn-xor.dump：学習後のネットワークの状態をダンプしたファイル

・テストは ./learn.py で実行
	＊最小二乗誤差を表示
		＋初期化直後と学習後の結果を表示
	＊テストデータは学習時に利用した数値をそのまま利用

・dump.py では、mnistのファイルダンプ用のメソッドを実装済み（次回の文字認識で利用予定）
