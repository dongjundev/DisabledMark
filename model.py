import cv2
import numpy as np

class main():

    def __init__(self):
        super().__init__()
        # Yolo 로드
        self.net = cv2.dnn.readNet("./yolo/yolov3_final.weights", "./yolo/yolov3.cfg")
        self.classes = []
        with open("./yolo/obj.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 웹캠 열기
        self.capture = cv2.VideoCapture(0)
        #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        #self.capture = cv2.resize(self.capture, None, fx=0.4, fy=0.4)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


        self.nextFrameSlot()

    def nextFrameSlot(self):
        #while cv2.waitKey(33) < 0:
        while True:
            try:
                _, frame = self.capture.read()
            except Exception as ex:
                print(ex)

            if frame is not None:
                frame = cv2.flip(frame, 1)

                # Detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)

                # 정보를 화면에 표시
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            #print(confidence)
                            # Object detected
                            center_x = int(detection[0] * 1280)
                            center_y = int(detection[1] * 720)
                            w = int(detection[2] * 1280)
                            h = int(detection[3] * 720)
                            # 좌표
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                #print(indexes)

                # 화면출력
                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        #label = 'disabled'
                        color = self.colors[i]
                        print(color)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)


                cv2.imshow("test", frame)

            if cv2.waitKey(1) == 27:
                self.capture.release()
                break  # esc to quit



if __name__ == '__main__':
     main()