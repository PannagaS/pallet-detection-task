import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image 
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import cv2
import os
import torch
# from IPython import embed
from ultralytics import YOLO
import numpy as np



class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        
        # Configure parameters
        self.declare_parameter('model_path', '/home/ws/models/best.engine')
        self.declare_parameter('save_dir', '/home/ws/predictions')
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('topic_to_subscribe', "/input_images")
        self.declare_parameter('mask_save_dir', '/home/ws/predictions/segmentation_masks')
        self.declare_parameter('save_predictions', 'False')
        self.declare_parameter('iou', 0.7)
        self.declare_parameter('half', "False")

        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
        topic_to_subscribe = self.get_parameter('topic_to_subscribe').get_parameter_value().string_value
        self.mask_save_dir = self.get_parameter('mask_save_dir').get_parameter_value().string_value
        self.save_predictions_flag = self.get_parameter('save_predictions').get_parameter_value().string_value
        self.iou = self.get_parameter('iou').get_parameter_value().double_value
        self.half = self.get_parameter('half').get_parameter_value().string_value

        self.save_predictions_flag = True if self.save_predictions_flag=="True" else False
        self.half = True if self.half == "True" else False

        # Subscriber 
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        self.subscriber_ = self.create_subscription(Image, topic_to_subscribe, self.listener_callback, qos_profile)
        self.bridge = CvBridge()

        # publisher to publish the yolo predictions to /all_detections  
        self.publisher_ = self.create_publisher(Image, '/all_detections', 20)

        # publisher to publish pallet detections to /pallet_detections  
        self.pallet_detection_publisher_ = self.create_publisher(Image, '/pallet_detections', 20)

        # publisher to publish ground detections to /ground_detections
        self.ground_detection_publisher_ = self.create_publisher(Image, '/ground_detections', 20)

        # publisher to publish ground segments to /ground_segmentmask 
        self.ground_segment_publisher_ = self.create_publisher(Image, '/ground_segmentmask', 20)

        # publisher to publish pallet segments to /pallet_segmentmask
        self.pallet_segment_publisher_ = self.create_publisher(Image, '/pallet_segmentmask', 20)

        self.model = YOLO(model_path, task='segment')
        self.image_count = 0
        if self.save_predictions_flag:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)   
            # os.makedirs(self.mask_save_dir, exist_ok=True)
            
            self.all_predictions_dir = os.path.join(save_dir, 'all_predictions') # all yolo predictions
            self.segment_mask_dir = os.path.join(save_dir, 'segment_mask') # to save segmentation masks
            self.detections_dir = os.path.join(save_dir, 'detections') # to extract and paint bboxes (redundant imo)
            os.makedirs(self.all_predictions_dir)
            os.makedirs(os.path.join(self.segment_mask_dir, 'class_0'))
            os.makedirs(os.path.join(self.segment_mask_dir, 'class_1'))
            os.makedirs(os.path.join(self.detections_dir, 'class_0'))
            os.makedirs(os.path.join(self.detections_dir, 'class_1'))
                
        self.model_ready = False
        
        
        self.get_logger().info("Loading model...")
        self.model = None
        self.load_model_async(model_path)
        self.ready_publisher_ = self.create_publisher(Bool, 'model_ready', 10)
        
    def load_model_async(self, model_path):
        """Load the model asynchronously and set the model_ready flag."""
        def load_model():
            try:
                self.model = YOLO(model_path, task='segment')
                self.model_ready = True
                self.ready_publisher_.publish(Bool(data=True))
                self.get_logger().info("Model loaded and ready.")
            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
        
         
        import threading
        threading.Thread(target=load_model).start()


    def listener_callback(self, msg):
        if not self.model_ready:
            self.get_logger().warn("Model is not ready yet. Skipping message.")
            return
        
        
        try:
            results = None
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if image is not None:
                # Predict for received image
                results = self.model.predict(image, conf=self.confidence, iou=self.iou, half=self.half)
                
                
                predicted_image = results[0].plot()

                class_1_masked_image = image.copy()
                class_0_masked_image = image.copy()

                class_1_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                class_0_mask = np.zeros(image.shape[:2], dtype=np.uint8)

                for result in results:
                    boxes = result.boxes  # Bounding box objects
                    masks = result.masks  # Segmentation mask objects
                    
                    if boxes is not None and masks is not None:
                        for i, (box, mask) in enumerate(zip(boxes, masks)):
                            cls = int(box.cls)
                            conf = float(box.conf)
                            xyxy = box.xyxy[0].tolist()
                            
                            self.get_logger().info(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")
                            
                            mask_np = mask.data.cpu().numpy().squeeze()
                            if mask_np.shape[:2] != image.shape[:2]:
                                mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                            
                            binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
                            
                            if cls == 1:  # Pallet
                                class_1_mask = cv2.bitwise_or(class_1_mask, binary_mask)
                                
                                # Create overlay for alpha blending
                                overlay = class_1_masked_image.copy()
                                cv2.rectangle(overlay, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), -1)
                                
                                # Apply alpha blending
                                alpha = 0.3
                                cv2.addWeighted(overlay[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], 
                                                alpha, 
                                                class_1_masked_image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], 
                                                1 - alpha, 
                                                0, 
                                                class_1_masked_image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])])
                                
                                # Draw bounding box and label
                                cv2.rectangle(class_1_masked_image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                                label = f"Pallet: {conf:.2f}"
                                cv2.putText(class_1_masked_image, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                            
                            elif cls == 0:  # Ground
                                class_0_mask = cv2.bitwise_or(class_0_mask, binary_mask)
                                
                                # Create overlay for alpha blending
                                overlay = class_0_masked_image.copy()
                                cv2.rectangle(overlay, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), -1)
                                
                                # Apply alpha blending
                                alpha = 0.3
                                cv2.addWeighted(overlay[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], 
                                                alpha, 
                                                class_0_masked_image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], 
                                                1 - alpha, 
                                                0, 
                                                class_0_masked_image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])])
                                
                                # Draw bounding box and label
                                cv2.rectangle(class_0_masked_image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                                label = f"Class 0: {conf:.2f}"
                                cv2.putText(class_0_masked_image, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                    if self.save_predictions_flag:
                        # Save class 1 (pallet) mask
                        class_1_mask_path = os.path.join(self.segment_mask_dir, 'class_1', f"pallet_mask_{self.image_count}.png")
                        cv2.imwrite(class_1_mask_path, class_1_mask)
                        # self.pallet_segment_publisher_.publish(self.bridge.cv2_to_imgmsg(class_1_mask, encoding="bgr8"))
                        self.get_logger().info(f"Saved pallet mask: {class_1_mask_path}")
                        # class_1_mask = self.bridge.imgmsg_to_cv2(class_1_mask, encoding="bgr8")

                        # Save class 0 mask
                        class_0_mask_path = os.path.join(self.segment_mask_dir, 'class_0', f"ground_mask_{self.image_count}.png")
                        cv2.imwrite(class_0_mask_path, class_0_mask)
                        # self.ground_segment_publisher_.publish(self.bridge.cv2_to_imgmsg(class_0_mask, encoding="bgr8"))
                        self.get_logger().info(f"Saved class 0 mask: {class_0_mask_path}")
                        # class_0_mask = self.bridge.imgmsg_to_cv2(class_0_mask, encoding="bgr8")

                        # Save masked images with bounding boxes
                        class_1_bbox = os.path.join(self.detections_dir, 'class_1', f"pallets_detected_{self.image_count}.png")
                        cv2.imwrite(class_1_bbox, class_1_masked_image)
                        # self.pallet_detection_publisher_.publish(self.bridge.cv2_to_imgmsg(class_1_masked_image, encoding="bgr8"))
                        self.get_logger().info(f"Saved bbox for pallets: {class_1_bbox}")
                        # class_1_masked_image = self.bridge.imgmsg_to_cv2(class_1_masked_image, encoding="bgr8")

                        class_0_bbox = os.path.join(self.detections_dir, 'class_0', f"ground_detected_{self.image_count}.png")
                        cv2.imwrite(class_0_bbox, class_0_masked_image)
                        # self.ground_detection_publisher_.publish(self.bridge.cv2_to_imgmsg(class_0_masked_image, encoding="bgr8"))
                        self.get_logger().info(f"Saved bbox for class 0: {class_0_bbox}")
                        # class_0_masked_image = self.bridge.imgmsg_to_cv2(class_0_masked_image, encoding="bgr8")
    

                self.pallet_segment_publisher_.publish(self.bridge.cv2_to_imgmsg(class_1_mask, encoding="mono8"))
                self.ground_segment_publisher_.publish(self.bridge.cv2_to_imgmsg(class_0_mask, encoding="mono8"))

                self.pallet_detection_publisher_.publish(self.bridge.cv2_to_imgmsg(class_1_masked_image, encoding="bgr8"))
                self.ground_detection_publisher_.publish(self.bridge.cv2_to_imgmsg(class_0_masked_image, encoding="bgr8"))
                
                if self.save_predictions_flag:
                    # Save the overall predicted image (with all detections)
                    file_path = os.path.join(self.all_predictions_dir, f"prediction_{self.image_count}.png")
                    cv2.imwrite(file_path, predicted_image)
                    self.get_logger().info(f"Saved predicted image: {file_path}")

                self.image_count += 1

                
                # convert the predicted image from cv2 format to msg (ROS format)
                predicted_image_msg = self.bridge.cv2_to_imgmsg(predicted_image, encoding="bgr8")
                self.publisher_.publish(predicted_image_msg)
                self.get_logger().info(f"Predicted image #{self.image_count} published.")

               

            


        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
