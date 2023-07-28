import os
import sys
from tabnanny import check
import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import glob
import config

# === config.py へ thresholdの値記述
threshold = config.threshold
tmp_info= config.pass_info
mode= config.mode  # 1 : on , 2 : OFF


face_locations = []
face_encodings = []

# === ディレクトリ下の画像を取得 
# image_paths = glob.glob('image/*')
image_paths = glob.glob('image_jp/*')
for img_val in image_paths:
    print("登録 画像ファイル名 :::" + img_val)


image_paths.sort()
known_face_encodings = []
known_face_names = []

checkd_face = []

# delimiter = "\\" # Windows用 (\記号を２つ書く)
# delimiter = "/"  # Mac/Linux用

for image_path in image_paths:
    # im_name = image_path.split(delimiter)[-1].split('.')[0]
    im_name = os.path.basename(image_path).split('.')[0]    # os.path.basename ファイル名の取得
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(im_name)


video_capture = cv2.VideoCapture(0)

#================== パスワードチェック
def check_password(name):
    if name in checkd_face:
        return
    
    emp_pw = input(name + "さんのパスワードを入力してください。")
    
    if tmp_info[name] == emp_pw:
        print("ログイン OK")
        checkd_face.append(name)
    else:
        print("パスワードが間違っています。")


#================== 認識率の計算
def calculate_recognition_rate(matches):
    num_matches = sum(matches)
    total_faces = len(matches)
    recognition_rate = num_matches / total_faces
    return recognition_rate


#================== main 処理
def main(): 
    
	print("=== main 処理 開始 ===")
 
	while True:
     
  
	    # ビデオの単一フレームを取得
		_, frame = video_capture.read()

		# ビデオの現在のフレーム内のすべての顔に対してその位置情報を検索
		face_locations = face_recognition.face_locations(frame)
		# 顔の位置情報からエンコードを生成
		face_encodings = face_recognition.face_encodings(frame, face_locations)

		for face_encoding in face_encodings:
			# 顔が登録済みの顔と一致するか確認
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding, threshold)
			name = "Unknown"

			# カメラ画像と最も近い登録画像を見つける
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]
    
			# 認識率を計算
			recognition_rate = calculate_recognition_rate(matches)
			recognition_rate_str = f"Recognition Rate: {recognition_rate:.2%}"

		# 位置情報の表示
		for (top, right, bottom, left) in face_locations:
      
        	# 顔領域に枠を描画 　色：緑
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 128, 0), 2)

			# 枠の下に名前を表示　色：緑
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 128, 0), cv2.FILLED)
			#font = cv2.FONT_HERSHEY_DUPLEX
			#cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
   
			# === 日本語表示 ===
			fontpath = 'meiryo.ttc'
			font = ImageFont.truetype(fontpath, 32)
			img_pil = Image.fromarray(frame)
			draw = ImageDraw.Draw(img_pil)
			position = (left + 6, bottom - 40)
			# drawにテキストを記載
			draw.text(position, name, font=font, fill=(255,255,255,0))
			frame = np.array(img_pil)
   
   
			# カメラ画像上に認識率を表示　色：赤
			cv2.putText(frame, recognition_rate_str, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
   
			# === 本人認証 ===
			if mode == 1 and name != "Unknown":
				check_password(name)
   
		
		# 結果をビデオに表示
		cv2.imshow('Video', frame)

		# t キーで終了
		if cv2.waitKey(1) & 0xFF == ord('t'):
			
			print("=== main 処理 終了 ===")
			# ウェブカメラへの操作を開放
			video_capture.release()
			cv2.destroyAllWindows()
			break


if __name__ == "__main__":
    main()


# ウェブカメラへの操作を開放
video_capture.release()
cv2.destroyAllWindows()
