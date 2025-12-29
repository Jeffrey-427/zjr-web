main.py文件是大作业的全部代码
submission.csv是模型在系统中获得最高分数的输出测试集
model/best.pt是取得最高分数的模型
在main.py文件的第36行，当ONLY_PREDICT = False时，代码从头开始训练模型并输出预测结果；当ONLY_PREDICT = True时，代码跳过模型训练阶段，直接用我训练好的最佳模型（也就是model/best.pt）跑出预测结果以及输出对应的测试集结果。（默认ONLY_PREDICT = False。如果想用我跑出来的最佳模型输出预测结果，需要把best.pt文件放在代码的同目录下。）
