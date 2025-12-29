# 项目说明

## 文件结构

### main.py
大作业的全部代码文件。

### submission.csv
模型在系统中获得最高分数的输出测试集结果。

### model/best.pt
取得最高分数的模型文件。

## 运行模式

在 `main.py` 文件的第36行，通过 `ONLY_PREDICT` 参数控制运行模式：

- **`ONLY_PREDICT = False`（默认）**  
  代码从头开始训练模型并输出预测结果。

- **`ONLY_PREDICT = True`**  
  代码跳过模型训练阶段，直接使用训练好的最佳模型（`model/best.pt`）运行预测并输出对应的测试集结果。

## 使用说明

如果想使用已训练好的最佳模型输出预测结果：

1. 将 `best.pt` 文件放在代码的同目录下
2. 设置 `ONLY_PREDICT = True`
3. 运行代码
