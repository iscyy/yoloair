当使用 ultralytics\cfg\models\cfg2024\YOLOv8-Loss\Focaler 文件下的内容时

需要将ultralytics\utils\NewLoss\iouloss.py文件里面的
代码
Focaler = False
改成
Focaler = True
即可