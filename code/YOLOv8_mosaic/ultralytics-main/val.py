# # '''
# # 实例分割验证
# # '''
# from ultralytics import YOLO
#
# if __name__=="__main__":
#     # Load a model
#     # model = YOLO('yolov8n-seg.pt')  # 加载官方的模型权重作评估
#     model = YOLO('./runs/segment/train/weights/best.pt') # 加载自定义的模型权重作评估
#
#    	# 评估
#     metrics = model.val()  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
#     # 如果你要在新的数据集上测试你的训练结果的话，需要将你的数据集绝对路径传进去，例如：
#     metrics = model.val(data='./ultralytics/cfg/datasets/human_body-seg.yaml')
#     print(f"mAP@0.5:0.95: \n{metrics.box.map}\n")  # map50-95
#     print(f"mAP@0.5: \n{metrics.box.map50}\n")  # map50
#     print(f"每个类别的mAP@0.5:0.95: \n{metrics.box.maps}\n")  # 包含每个类别的map50-95列表
#
# 	# # Accessing AP75 for each category
# 	# ap75_each_category = metrics.box.maps[:, 5]  # 利用maps矩阵可以得到AP75
# 	# print(ap75_each_category)