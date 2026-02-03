;; robot-score-0.l  (EusLisp)
;; 目的:
;; - 指定した右腕関節角のシーケンスを実行
;; - 送信は必ず *jedy* の角度ベクトル経由で行う

(format t "~%[ROBOT] score=0: 右腕関節角のシーケンスを実行します。~%")

;; ------------------------------------------------------------
;; Helper: 今動いているときだけキャンセルして、その後に停止を待つ
;; ------------------------------------------------------------
(defun safe-cancel-and-wait (&optional (wait-time 0.1))
  (when (send *ri* :interpolatingp)
    (send *ri* :cancel-angle-vector)
    (unix:sleep wait-time)
    (when (send *ri* :interpolatingp)
      (send *ri* :wait-interpolation))))

;; ------------------------------------------------------------
;; Helper: 角度送信 + 完了待ち（preempt回避の基本）
;; ------------------------------------------------------------
(defun send-av-and-wait (msec)
  (send *ri* :angle-vector (send *jedy* :angle-vector) msec)
  (send *ri* :wait-interpolation))

;; サーボON
(send *ri* :servo-on)

;; 走っている軌道があれば止める
(safe-cancel-and-wait)

;; reset-pose をモデル側に適用し、実機へ送る
(send *jedy* :reset-pose)
(send-av-and-wait 1000)

;; 1) rarm_joint0 を -90deg
(send *jedy* :rarm_joint0 :joint-angle -20)

(send *jedy* :rarm_joint5 :joint-angle -55)

(send *jedy* :rarm_gripper_joint :joint-angle 89.9)

(send-av-and-wait 1000)

;; 2) rarm_joint5 を 0deg
(send *jedy* :rarm_joint5 :joint-angle 20)

;; 3) rarm_joint0 を -35deg
(send *jedy* :rarm_joint0 :joint-angle -5)

;; 4) rarm_gripper_joint を -100deg
(send *jedy* :rarm_gripper_joint :joint-angle -28.6478)

;; 送信
(send-av-and-wait 1000)

(format t "~%[ROBOT] done.~%")
