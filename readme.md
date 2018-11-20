#配置
在config里头对tensorflow serving的地址和端口进行配置

#文件用途
mnist_deep_train  是将tf的tutorial里头的mnist_deep.py改造成可以save model，output summary还可以export model的训练脚本。
mul_imgs_reg_demo.py 采用图形界面选择多个图片，点击识别按钮进行识别，还支持翻页

#启动serving服务
docker run -p 8500:8500 -p 8501:8501 \
  --mount type=bind,source=/data/dl_models/tensorflow,target=/models \
  -e MODEL_NAME=mnist_deep_demo \
  -t tensorflow/serving
  
##注意 
1. 8501是restful api用的端口，8500是grpc的端口，本程序用的grpc方式
2. 代码里头的request.model_spec.name = 'mnist_deep_demo'  
    这里的名字要跟serving的MODEL_NAME一一对应，否则会找不到
3. 实际模型所在的host文件夹为/data/dl_models/tensorflow/mnist_deep_demo/{version}

#运行程序
运行mul_imgs_reg_demo.py即可

#code解析
tensorflow serving服务提供两种访问方式：
1. grpc 方式。采用此方式时，用predict.py里头的do_inference_grpc函数（重命名成do_inference）
2. restful api 方式。采用此方式时，现有代码不用动。
为了验证服务部署是否正确，可以简单采用以下命令进行访问：
curl -i -d '{"signature_name": "predict_images", "instances": [{"images": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003921568859368563, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007843137718737125, 0.0, 0.16862745583057404, 0.10980392247438431, 0.0, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0117647061124444, 0.0, 0.3803921639919281, 0.1764705926179886, 0.003921568859368563, 0.0117647061124444, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003921568859368563, 0.003921568859368563, 0.40392157435417175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0470588244497776, 0.4745098054409027, 0.14901961386203766, 0.13333334028720856, 0.13333334028720856, 0.13333334028720856, 0.13333334028720856, 0.13333334028720856, 0.13333334028720856, 0.13333334028720856, 0.0941176488995552, 0.06666667014360428, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007843137718737125, 0.0, 0.16862745583057404, 0.4745098054409027, 0.5098039507865906, 0.5411764979362488, 0.5411764979362488, 0.5372549295425415, 0.5411764979362488, 0.5411764979362488, 0.5372549295425415, 0.5372549295425415, 0.5568627715110779, 0.48627451062202454, 0.0117647061124444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0117647061124444, 0.0, 0.3019607961177826, 0.1921568661928177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0117647061124444, 0.0, 0.3490196168422699, 0.14901961386203766, 0.0117647061124444, 0.01568627543747425, 0.0117647061124444, 0.0117647061124444, 0.0117647061124444, 0.0117647061124444, 0.0117647061124444, 0.0117647061124444, 0.007843137718737125, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007843137718737125, 0.0, 0.37254902720451355, 0.0941176488995552, 0.007843137718737125, 0.007843137718737125, 0.007843137718737125, 0.003921568859368563, 0.007843137718737125, 0.0117647061124444, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007843137718737125, 0.0, 0.40392157435417175, 0.06666667014360428, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0117647061124444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0117647061124444, 0.0, 0.3607843220233917, 0.5882353186607361, 0.545098066329956, 0.5607843399047852, 0.5607843399047852, 0.5568627715110779, 0.5490196347236633, 0.4117647111415863, 0.0313725508749485, 0.0, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667014360428, 0.09019608050584793, 0.08627451211214066, 0.09019608050584793, 0.09019608050584793, 0.11764705926179886, 0.3764705955982208, 0.615686297416687, 0.30980393290519714, 0.0, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0784313753247261, 0.5411764979362488, 0.23137255012989044, 0.0, 0.007843137718737125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.003921568859368563, 0.019607843831181526, 0.0, 0.11372549086809158, 0.3294117748737335, 0.0, 0.0117647061124444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0117647061124444, 0.007843137718737125, 0.007843137718737125, 0.007843137718737125, 0.0117647061124444, 0.01568627543747425, 0.007843137718737125, 0.0, 0.3607843220233917, 0.2980392277240753, 0.0, 0.0117647061124444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03921568766236305, 0.062745101749897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3960784375667572, 0.5333333611488342, 0.03529411926865578, 0.007843137718737125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003921568859368563, 0.003921568859368563, 0.08235294371843338, 0.615686297416687, 0.40392157435417175, 0.20000000298023224, 0.1921568661928177, 0.24705882370471954, 0.3333333432674408, 0.42352941632270813, 0.5921568870544434, 0.4313725531101227, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04313725605607033, 0.364705890417099, 0.49803921580314636, 0.4941176474094391, 0.4627451002597809, 0.3921568691730499, 0.3019607961177826, 0.09019608050584793, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0117647061124444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003921568859368563, 0.0117647061124444, 0.0117647061124444, 0.0117647061124444, 0.01568627543747425, 0.01568627543747425, 0.0117647061124444, 0.003921568859368563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "keep_prob": 1.0}]}' -X POST http://localhost:8501/v1/models/mnist_deep_demo:predict

结果为：

HTTP/1.1 100 Continue

HTTP/1.1 200 OK
Content-Type: application/json
Date: Wed, 14 Nov 2018 18:00:39 GMT
Content-Length: 156

{
    "predictions": [[1.49083e-07, 1.27124e-07, 5.70308e-05, 0.000327526, 8.19566e-08, 0.999281, 1.02937e-06, 7.05844e-05, 5.93887e-06, 0.00025685]
    ]
}

此外，还可以 
curl http://localhost:8501/v1/models/mnist_deep_demo          查看模型状态
curl http://localhost:8501/v1/models/mnist_deep_demo/metadata 查看模型参数