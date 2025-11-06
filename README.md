# Local-multilingual-real-time-speech-recognition-and-real-time-subtitle-translation
在端侧（RKxxxx、MTKxxx等系列芯片）Android设备上实现多语言本地实时语音识别与翻译、采用的主要模型是 ASR: whisper  Translate: Helsinki

1 在RK芯片上移植算法流程

（1） 先在python环境下，分别导出对应的onnx格式的 encoder和decoder（导出的时候注意和RK平台的适配，避免动态shape） 

（2） 在Python环境下用导出的onnx格式的encoder和decoder进行协同工作，测试完成编码器和解码器的协同工作，验证导出的模型的可用性 

（3） 验证完第二步之后，再继续在linux机器的python环境下，完成encoder.onnx和decoder.onnx 格式到rknn格式的转换 

（4） 根据XXXX模型的官方python推理代码、复刻编写Android环境下的C++推理代码。

    用输入的简单测试token完成XXXX模型(encoder.rknn和decoder.rknn)在RKXXX芯片上的协同推理工作（编码+解码），验证rknn格式模型以及C++版推理代码的可用性 

（5） 实现tokenizer，C++端实现tokenizer

    根据导出的vocab.txt，实现词表的映射，测试encoder能够正常工作，测试decoder能否正常工作，完成完整的C++端推理流程