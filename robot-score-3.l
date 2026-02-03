;; robot-score-3-fixed-seq.l  (EusLisp)
;; 目的:
;; - :angle-vector 連発による preempt 回避（毎回 wait）
;; - 「no goal exists」回避（goal無いときは wait-for-result しない）
;; - IK失敗時は実機へ送らず中断（安全）

(format t "~%[ROBOT] score=3: 指定した右腕IKシーケンスを実行します。~%")

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
;; Helper: IK 1回（XYZ自由） -> 成功なら実機送信、失敗なら即中断
;; ------------------------------------------------------------
(defun rarm-ik-step-free (pos rpy &key (msec 3000) rthre)
  (let ((ik-ok
          (if rthre
              (send *jedy* :rarm :inverse-kinematics
                    (make-coords :pos pos :rpy rpy)
                    :rthre rthre
                    :debug-view :no-message)
              (send *jedy* :rarm :inverse-kinematics
                    (make-coords :pos pos :rpy rpy)
                    :debug-view :no-message))))
    (unless ik-ok
      (format t "~%[WARN] IK failed at pos=~a rpy=~a. Motion is aborted for safety.~%"
              pos rpy)
      (return-from rarm-ik-step-free nil))
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

;; 0.5) 右腕先端の高さを下げる（ワールド座標系でZ方向へ移動）
(send *jedy* :rarm :move-end-pos (float-vector 0 0 -30) :world
      :rotation-axis nil)
(send-av-and-wait 2000)

;; 1) 指定された IK シーケンスを順番に実行
(let ((steps
        (list
         ;; (pos . rpy) のペア
         (cons (float-vector 190 -60 -20)  (list (deg2rad 50) (deg2rad 0) (deg2rad 0)))
         (cons (float-vector 190 -60 15)  (list (deg2rad 50) (deg2rad 0) (deg2rad 0)))
         (cons (float-vector 190 -140 0)   (list (deg2rad 10) (deg2rad 0) (deg2rad 0)))
         (cons (float-vector 155 -70 -20)  (list (deg2rad 50) (deg2rad 0) (deg2rad 0)))
         (cons (float-vector 155 -130 -20) (list (deg2rad 10) (deg2rad 0) (deg2rad 0)))
         (cons (float-vector 120 -50 -10)  (list (deg2rad 50) (deg2rad 0) (deg2rad 0))))))

  (dolist (st steps)
    (let ((pos (car st))
          (rpy (cdr st)))
      (unless
          (if (equal pos (float-vector 190 -60 15))
              (rarm-ik-step-free pos rpy
                                 :rthre nil
                                 :msec 3000)
              (rarm-ik-step pos rpy
                            :translation-axis :z
                            :rthre (deg2rad 15)
                            :msec 3000))
        ;; 途中失敗したらここで終了（実機へは失敗姿勢を送っていない）
        (return))))

  (format t "~%[ROBOT] sequence done.~%"))

(format t "~%[ROBOT] done.~%")
