;; robot-score-1.l  (EusLisp)
;; 目的:
;; - :angle-vector 連発による preempt 回避（毎回 wait）
;; - 「no goal exists」回避（goal無いときは wait-for-result しない）
;; - IK失敗時は実機へ送らず中断（安全）

(format t "~%[ROBOT] score=1: 指定した右腕IKシーケンスを実行します。~%")

;; ------------------------------------------------------------
;; Helper: 今動いているときだけキャンセルして、その後に停止を待つ
;; ------------------------------------------------------------
(defun safe-cancel-and-wait (&optional (wait-time 0.1))
  (when (send *ri* :interpolatingp)
    (send *ri* :cancel-angle-vector)
    (unix:usleep (round (* wait-time 1000000)))
    (when (send *ri* :interpolatingp)
      (send *ri* :wait-interpolation))))

;; ------------------------------------------------------------
;; Helper: 角度送信 + 完了待ち（preempt回避の基本）
;; ------------------------------------------------------------
(defun send-av-and-wait (msec)
  (send *ri* :angle-vector (send *jedy* :angle-vector) msec)
  (send *ri* :wait-interpolation))

;; ------------------------------------------------------------
;; Helper: IK 1回 -> 成功なら実機送信、失敗なら即中断
;; ------------------------------------------------------------
(defun rarm-ik-step (pos rpy &key (msec 3000) (rthre (deg2rad 15)) (translation-axis :z))
  (let ((ik-ok
          (send *jedy* :rarm :inverse-kinematics
                (make-coords :pos pos :rpy rpy)
                :translation-axis translation-axis
                :rthre rthre
                :debug-view :no-message)))
    (unless ik-ok
      (format t "~%[WARN] IK failed at pos=~a rpy=~a. Motion is aborted for safety.~%"
              pos rpy)
      ;; 呼び出し元に失敗を返す
      (return-from rarm-ik-step nil))
    ;; IK後姿勢を実機へ送って完了待ち
    (send-av-and-wait msec)
    t))

;; ------------------------------------------------------------
;; Main
;; ------------------------------------------------------------
(send *ri* :servo-on)

;; 走っている軌道があれば止める（無い時は何もしない）
(safe-cancel-and-wait)

;; 0) 指定の angle-vector をモデル側に適用し、実機へ送る
(send *jedy* :angle-vector #f(90.0 -4.0 -30.0 -100.0 -3.0 -88.0 -1.0 89.9 -90.0 4.0 30.0 -100.0 -88.0 -6.0 89.0 -1.0 0.0 0.0))
(send-av-and-wait 3000)

;; 1) 指定された IK シーケンスを順番に実行
(let ((steps
        (list
         ;; (pos . rpy) のペア
         (cons (float-vector 190 -100 -10)  (list (deg2rad 30) (deg2rad 0) (deg2rad 0)))
         (cons (float-vector 120 -100 -10) (list (deg2rad 30) (deg2rad 0) (deg2rad 0))))))

  (dolist (st steps)
    (unless (rarm-ik-step (car st) (cdr st)
                          :translation-axis :z
                          :rthre (deg2rad 15)
                          :msec 3000)
      ;; 途中失敗したらここで終了（実機へは失敗姿勢を送っていない）
      (return)))

  (format t "~%[ROBOT] sequence done.~%"))

(format t "~%[ROBOT] done.~%")
