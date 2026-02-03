;; haiku_robot.l（簡易版）
;; - ROS1 + roseus 前提
;; - すでに jedy-interface.l がロードされ、(jedy-init) 済みで *jedy* や *ri* が定義されている前提
;; - このファイルを (load "haiku_robot.l") してから (haiku-init) を呼ぶ

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; haiku_msgs を読み込む
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(ros::load-ros-manifest "haiku_msgs")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; グローバル変数
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defvar *haiku-subscriber* nil)
(defvar *lineart-subscriber* nil)
(defvar *lineart-z* -30.0)
(defvar *lineart-msec* 1500)
(defvar *lineart-x-min* 120.0)
(defvar *lineart-x-max* 190.0)
(defvar *lineart-y-min* -150.0)
(defvar *lineart-y-max* -50.0)
(defvar *lineart-z-min* -10.0)
(defvar *lineart-z-max* 0.0)
(defvar *lineart-roll-min* 10.0)
(defvar *lineart-roll-max* 50.0)
(defvar *lineart-home-av* #f(40.0 -4.0 -30.0 -100.0 -3.0 -70.0 -1.0 89.9 -90.0 4.0 30.0 -100.0 -88.0 -6.0 89.0 -1.0 0.0 0.0))
(defvar *lineart-use-joint* nil)
(defvar *penair-path* "/home/mech-user/haiku_pos/npenair.csv")
(defvar *pengnd-path* "/home/mech-user/haiku_pos/npengnd.csv")
(defvar *penair-table* nil)
(defvar *pengnd-table* nil)
(defvar *lineart-joint-subscriber* nil)
(defvar *lineart-joint-wait-timeout* 8.0)
(defvar *lineart-joint-busy* nil)
(defvar *lineart-joint-task* nil)
(defvar *lineart-joint-debug* t)
(defvar *lineart-joint-mode* 'interp)
(defvar *lineart-joint-snap-subscriber* nil)
(defvar *lineart-grid-step* 10.0)
(defvar *lineart-snap-threshold* 5.0)
(defvar *lineart-seg-threshold* 30.0)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; コールバック関数
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun haiku-result-cb (msg)
  (let* ((score   (send msg :score))
         (comment (send msg :comment))
         (revised (send msg :revised))
         (original (send msg :original)))
    (format t "~%[HAIKU] 俳句評価を受信しました:~%")
    (format t "  score   : ~A~%" score)
    (format t "  original: ~A~%" original)
    (format t "  comment : ~A~%" comment)
    (format t "  revised : ~A~%~%" revised)

    ;; score に応じて動作スクリプトをロード
    (case score
      (0
       (format t "[HAIKU] score=0 → robot-score-0.l をロードします。~%")
       (if (probe-file "robot-score-0.l")
           (load "robot-score-0.l")
           (format t "[HAIKU] robot-score-0.l が見つかりません。~%")))
      (1
       (format t "[HAIKU] score=1 → robot-score-1.l をロードします。~%")
       (if (probe-file "robot-score-1.l")
           (load "robot-score-1.l")
           (format t "[HAIKU] robot-score-1.l が見つかりません。~%")))
      (2
       (format t "[HAIKU] score=2 → robot-score-2.l をロードします。~%")
       (if (probe-file "robot-score-2.l")
           (load "robot-score-2.l")
           (format t "[HAIKU] robot-score-2.l が見つかりません。~%")))
      (3
       (format t "[HAIKU] score=3 → robot-score-3.l をロードします。~%")
       (if (probe-file "robot-score-3.l")
           (load "robot-score-3.l")
           (format t "[HAIKU] robot-score-3.l が見つかりません。~%")))
      (4
       (format t "[HAIKU] score=4 → reset-pose を実行します。~%")
       (send *ri* :servo-on)
       (lineart-safe-cancel-and-wait)
       (send *jedy* :reset-pose)
       (send *ri* :angle-vector (send *jedy* :angle-vector) 1000)
       (send *ri* :wait-interpolation))
      (5
       (format t "[HAIKU] score=5 → 指定角度姿勢を実行します。~%")
       (send *ri* :servo-on)
       (lineart-safe-cancel-and-wait)
       (send *jedy* :angle-vector #f(90.0 -4.0 -30.0 -100.0 -3.0 -88.0 -1.0 89.9 -90.0 4.0 30.0 -100.0 -88.0 -6.0 89.0 -1.0 0.0 0.0))
       (send *ri* :angle-vector (send *jedy* :angle-vector) 1000)
       (send *ri* :wait-interpolation))
      (t
       (format t "[HAIKU] 想定外の score=~A のため何もしません。~%" score)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; LineArt 受信 -> 動作
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun lineart-xy-to-polylines (xy counts)
  (let ((idx 0)
        (result '()))
    (dotimes (ci (length counts))
      (let* ((c (elt counts ci))
             (pts '()))
        (dotimes (i c)
          (when (< (+ idx 1) (length xy))
            (setq pts (append pts (list (list (elt xy idx)
                                              (elt xy (+ idx 1)))))))
          (setq idx (+ idx 2)))
        (when (>= (length pts) 2)
          (setq result (append result (list pts))))))
    result))

;; ------------------------------------------------------------
;; Helper: 今動いているときだけキャンセルして、その後に停止を待つ
;; ------------------------------------------------------------
(defun lineart-safe-cancel-and-wait (&optional (wait-time 0.1))
  (when (send *ri* :interpolatingp)
    (send *ri* :cancel-angle-vector)
    (unix:usleep (round (* wait-time 1000000)))
    (when (send *ri* :interpolatingp)
      (send *ri* :wait-interpolation))))

;; ------------------------------------------------------------
;; Helper: 角度送信 + 完了待ち
;; ------------------------------------------------------------
(defun lineart-send-av-and-wait (msec)
  (if *lineart-use-joint*
      (progn
        (lineart-safe-cancel-and-wait)
        (send *ri* :angle-vector (send *jedy* :angle-vector) msec)
        (send *ri* :wait-interpolation))
      (progn
        (send *ri* :angle-vector (send *jedy* :angle-vector) msec)
        (send *ri* :wait-interpolation))))

(defun lineart-wait-interpolation-timeout (timeout-sec)
  (let ((elapsed 0.0)
        (step 0.1)
        (tick 0))
    (while (and (send *ri* :interpolatingp) (< elapsed timeout-sec))
      (ros::spin-once)
      (unix:usleep (round (* step 1000000)))
      (setq elapsed (+ elapsed step))
      (setq tick (+ tick 1))
      (when (and *lineart-joint-debug* (>= tick 10))
        (format t "~%[LINEART-JOINT] waiting... elapsed=~A sec.~%" elapsed)
        (finish-output)
        (setq tick 0)))
    (if (send *ri* :interpolatingp)
        nil
        t)))

(defun lineart-wait-start (&optional (timeout-sec 0.3))
  (let ((elapsed 0.0)
        (step 0.05))
    (while (and (not (send *ri* :interpolatingp)) (< elapsed timeout-sec))
      (ros::spin-once)
      (unix:usleep (round (* step 1000000)))
      (setq elapsed (+ elapsed step)))
    (when (and *lineart-joint-debug* (not (send *ri* :interpolatingp)))
      (format t "~%[LINEART-JOINT] interpolatingp did not start (elapsed=~A).~%" elapsed)
      (finish-output))
    (send *ri* :interpolatingp)))

(defun lineart-joint-refresh ()
  (when (and (boundp '*irtviewer*) *irtviewer*)
    (send *irtviewer* :draw-objects)
    (send *irtviewer* :viewer :viewsurface :flush))
  (finish-output))

(defun lineart-joint-run-task ()
  (let ((task *lineart-joint-task*))
    (when task
      (setq *lineart-joint-task* nil)
      (let ((polylines (nth 0 task))
            (width (nth 1 task))
            (height (nth 2 task))
            (mode (nth 3 task))
            (prev *lineart-use-joint*)
            (prev-mode *lineart-joint-mode*))
        (setq *lineart-joint-busy* t)
        (unwind-protect
            (progn
              (setq *lineart-use-joint* t)
              (setq *lineart-joint-mode* mode)
              (lineart-run polylines width height))
          (setq *lineart-use-joint* prev)
          (setq *lineart-joint-mode* prev-mode)
          (setq *lineart-joint-busy* nil))))))

(defun lineart-apply-point (x y z roll-deg)
  (let ((coords (make-coords :pos (float-vector x y z)
                             :rpy (list (deg2rad roll-deg) (deg2rad 0) (deg2rad 0)))))
    (let ((ik-ok
            (send *jedy* :rarm :inverse-kinematics
                  coords
                  :translation-axis :z
                  :rthre (deg2rad 10))))
      (unless ik-ok
        (format t "~%[LINEART] IK failed at x=~A y=~A z=~A roll=~A~%"
                x y z roll-deg)
        (return-from lineart-apply-point nil))
      (lineart-send-av-and-wait *lineart-msec*)
      t)))

(defun lineart-clamp (v vmin vmax)
  (min vmax (max vmin v)))

(defun split-csv-line (line)
  (unless (stringp line)
    (return-from split-csv-line '()))
  (let ((res '())
        (cur ""))
    (dotimes (i (length line))
      (let ((ch (char line i)))
        (if (eql ch #\,)
            (progn
              (setq res (append res (list cur)))
              (setq cur ""))
            (setq cur (concatenate string cur (string ch))))))
    (setq res (append res (list cur)))
    (mapcar #'(lambda (s) (string-right-trim '(#\Return #\Space #\Tab) s)) res)))

(defun parse-angle-cell (cell)
  (let ((s (if (stringp cell)
               (string-trim '(#\Space #\Tab #\Return) cell)
               "")))
    (if (string= s "")
        nil
        (if (and (>= (length s) 4)
                 (string= (subseq s 0 3) "#f(")
                 (eql (char s (- (length s) 1)) #\)))
            (let* ((inner (subseq s 3 (- (length s) 1)))
                   (list-str (concatenate string "(" inner ")"))
                   (lst (car (read-from-string list-str))))
              (coerce lst 'vector))
            (car (read-from-string s))))))

(defun plist-get (plist key)
  (let ((p plist))
    (while p
      (when (eql (car p) key)
        (return-from plist-get (cadr p)))
      (setq p (cddr p)))
    nil))

(defun plist-set (plist key value)
  (let ((p plist))
    (while p
      (when (eql (car p) key)
        (setf (cadr p) value)
        (return-from plist-set plist))
      (setq p (cddr p)))
    (append plist (list key value))))

(defun pen-cell-to-vector (cell)
  (cond
    ((null cell) nil)
    ((vectorp cell) cell)
    ((listp cell) (apply #'float-vector cell))
    (t (float-vector cell))))

(defun pen-table-coerce (table)
  (let* ((vals (plist-get table :vals))
         (fixed (mapcar
                 #'(lambda (row)
                     (mapcar #'pen-cell-to-vector row))
                 vals)))
    (plist-set table :vals fixed)))

(defun lineart-nearest-grid (vals v)
  (let ((best (car vals))
        (bd (abs (- v (car vals)))))
    (dolist (x (cdr vals))
      (let ((d (abs (- v x))))
        (when (< d bd)
          (setq bd d)
          (setq best x))))
    best))

(defun lineart-corner-points (xs ys x y)
  (let* ((bx (find-bracket xs x))
         (by (find-bracket ys y))
         (x0 (nth (nth 0 bx) xs))
         (x1 (nth (nth 1 bx) xs))
         (y0 (nth (nth 0 by) ys))
         (y1 (nth (nth 1 by) ys)))
    (list (list x0 y0) (list x1 y0) (list x0 y1) (list x1 y1))))

(defun lineart-snap-or-interp (table x y &optional (force-snap nil))
  (let* ((xs (plist-get table :xs))
         (ys (plist-get table :ys))
         (corners (lineart-corner-points xs ys x y))
         (min-d 1.0e9)
         (best (car corners)))
    (dolist (pt corners)
      (let* ((dx (- x (car pt)))
             (dy (- y (cadr pt)))
             (d (sqrt (+ (* dx dx) (* dy dy)))))
        (when (< d min-d)
          (setq min-d d)
          (setq best pt))))
    (if (or force-snap (>= min-d *lineart-snap-threshold*))
        (list 'interp x y)
        (list 'snap (car best) (cadr best)))))

(defun lineart-append-unique (out last x y)
  (if (or (null last)
          (> (abs (- x (car last))) 1.0e-6)
          (> (abs (- y (cadr last))) 1.0e-6))
      (list (append out (list (list x y))) (list x y))
      (list out last)))

(defun lineart-expand-with-grid (pts table)
  (let* ((xs (plist-get table :xs))
         (ys (plist-get table :ys))
         (out '())
         (last nil))
    (when pts
      (let ((res (lineart-append-unique out last (car (car pts)) (cadr (car pts)))))
        (setq out (car res))
        (setq last (cadr res))))
    (dotimes (i (- (length pts) 1))
      (let* ((p0 (nth i pts))
             (p1 (nth (+ i 1) pts))
             (x0 (car p0))
             (y0 (cadr p0))
             (x1 (car p1))
             (y1 (cadr p1))
             (dx (- x1 x0))
             (dy (- y1 y0))
             (dist (sqrt (+ (* dx dx) (* dy dy)))))
        (when (>= dist *lineart-seg-threshold*)
          (let* ((step *lineart-grid-step*)
                 (n (floor (/ dist step))))
            (dotimes (k n)
              (let* ((frac (/ (+ k 1.0) (+ n 1.0)))
                     (xi (+ x0 (* dx frac)))
                     (yi (+ y0 (* dy frac)))
                     (gx (lineart-nearest-grid xs xi))
                     (gy (lineart-nearest-grid ys yi))
                     (dd (sqrt (+ (* (- xi gx) (- xi gx))
                                  (* (- yi gy) (- yi gy))))))
                (when (<= dd *lineart-snap-threshold*)
                  (let ((res (lineart-append-unique out last gx gy)))
                    (setq out (car res))
                    (setq last (cadr res))))))))
        (let ((res (lineart-append-unique out last x1 y1)))
          (setq out (car res))
          (setq last (cadr res)))))
    out))

(defun load-pen-table (path)
  (let ((cache (concatenate string path ".lisp")))
    (when (probe-file cache)
      (with-open-file (s cache)
        (let ((table (read s nil nil)))
          (return-from load-pen-table (pen-table-coerce table))))))
  (with-open-file (s path)
    (let* ((header (split-csv-line (read-line s nil nil)))
           (ys (mapcar #'read-from-string (cdr header)))
           (xs '())
           (vals '()))
      (loop for line = (read-line s nil nil) while line do
        (when (and (stringp line) (> (length line) 0))
          (let* ((cells (split-csv-line line))
                 (xcell (and cells (car cells))))
            (when (and (stringp xcell) (not (string= xcell "")))
              (let* ((x (read-from-string xcell))
                     (row (mapcar #'parse-angle-cell (cdr cells))))
                (setq xs (append xs (list x)))
                (setq vals (append vals (list row))))))))
      (pen-table-coerce (list :xs xs :ys ys :vals vals)))))

(defun pen-table-valid-p (table)
  (let* ((xs (plist-get table :xs))
         (ys (plist-get table :ys))
         (vals (plist-get table :vals))
         (ok t)
         (vec-len nil))
    (when (or (null xs) (null ys) (null vals))
      (setq ok nil))
    (when ok
      (unless (= (length vals) (length xs))
        (setq ok nil)))
    (when ok
      (dolist (row vals)
        (unless (= (length row) (length ys))
          (setq ok nil))
        (dolist (cell row)
          (when (null cell) (setq ok nil))
          (when (and cell (vectorp cell))
            (if vec-len
                (unless (= (length cell) vec-len) (setq ok nil))
                (setq vec-len (length cell)))))))
    ok))

(defun ensure-pen-tables ()
  (unless *penair-table*
    (setq *penair-table* (load-pen-table *penair-path*)))
  (unless *pengnd-table*
    (setq *pengnd-table* (load-pen-table *pengnd-path*)))
  (and (pen-table-valid-p *penair-table*)
       (pen-table-valid-p *pengnd-table*)))

(defun find-bracket (vals v)
  (let* ((n (length vals))
         (v0 (car vals))
         (vn (car (last vals)))
         (desc (>= v0 vn)))
    (cond
      ((<= n 1) (list 0 0 0.0))
      (desc
       (if (>= v v0) (list 0 1 0.0)
           (if (<= v vn) (list (- n 2) (- n 1) 1.0)
               (let ((i 0))
                 (while (< i (- n 1))
                   (let ((a (nth i vals)) (b (nth (+ i 1) vals)))
                     (when (and (<= v a) (>= v b))
                       (let ((tx (/ (- v a) (- b a))))
                         (return-from find-bracket (list i (+ i 1) tx))))
                     (setq i (+ i 1))))))))
      (t
       (if (<= v v0) (list 0 1 0.0)
           (if (>= v vn) (list (- n 2) (- n 1) 1.0)
               (let ((i 0))
                 (while (< i (- n 1))
                   (let ((a (nth i vals)) (b (nth (+ i 1) vals)))
                     (when (and (>= v a) (<= v b))
                       (let ((tx (/ (- v a) (- b a))))
                         (return-from find-bracket (list i (+ i 1) tx))))
                     (setq i (+ i 1)))))))))))

(defun interp-angle (v00 v10 v01 v11 tx ty)
  (let* ((n (length v00))
         (out (make-array n :element-type 'float)))
    (dotimes (i n)
      (let* ((a (+ (* (aref v00 i) (- 1.0 tx)) (* (aref v10 i) tx)))
             (b (+ (* (aref v01 i) (- 1.0 tx)) (* (aref v11 i) tx)))
             (v (+ (* a (- 1.0 ty)) (* b ty))))
        (setf (aref out i) v)))
    out))

(defun table-angle-at (table x y)
  (let* ((xs (plist-get table :xs))
         (ys (plist-get table :ys))
         (vals (plist-get table :vals))
         (bx (find-bracket xs x))
         (by (find-bracket ys y))
         (i0 (nth 0 bx))
         (i1 (nth 1 bx))
         (tx (nth 2 bx))
         (j0 (nth 0 by))
         (j1 (nth 1 by))
         (ty (nth 2 by))
         (row0 (nth i0 vals))
         (row1 (nth i1 vals))
         (v00 (nth j0 row0))
         (v10 (nth j0 row1))
         (v01 (nth j1 row0))
         (v11 (nth j1 row1)))
    (interp-angle v00 v10 v01 v11 tx ty)))

(defun lineart-apply-angle (av)
  (let* ((out (if (vectorp av) (copy-seq av) (coerce av 'vector))))
    (when (> (length out) 6)
      (when (> (aref out 6) 89.9)
        (setf (aref out 6) 89.9)))
    (when (> (length out) 7)
      (setf (aref out 7) 89.9))
    (send *jedy* :angle-vector out)
    (lineart-send-av-and-wait *lineart-msec*)))

(defun lineart-compute-scale (width height polylines)
  (let* ((w (float width))
         (h (float height))
         (xrange (- *lineart-x-max* *lineart-x-min*))
         (yrange (- *lineart-y-max* *lineart-y-min*)))
    (if (or (<= w 0.0) (<= h 0.0))
        (list 1.0 0.0 0.0 nil)
        (let* ((xmin w)
               (xmax 0.0)
               (ymin h)
               (ymax 0.0))
          (dolist (pl polylines)
            (dolist (pt pl)
              (let ((x (float (car pt)))
                    (y (float (cadr pt))))
                (when (< x xmin) (setq xmin x))
                (when (> x xmax) (setq xmax x))
                (when (< y ymin) (setq ymin y))
                (when (> y ymax) (setq ymax y)))))
          (let* ((bw (max 1.0 (- xmax xmin)))
                 (bh (max 1.0 (- ymax ymin)))
                 (sx (/ xrange bw))
                 (sy (/ yrange bh))
                 (scale (min sx sy))
                 (xoff (/ (- xrange (* scale bw)) 2.0))
                 (yoff (/ (- yrange (* scale bh)) 2.0))
                 (sxr (/ xrange bh))
                 (syr (/ yrange bw))
                 (scale-r (min sxr syr))
                 (xoff-r (/ (- xrange (* scale-r bh)) 2.0))
                 (yoff-r (/ (- yrange (* scale-r bw)) 2.0)))
            (if (> scale-r scale)
                (list scale-r xoff-r yoff-r t xmin ymin xmax ymax)
                (list scale xoff yoff nil xmin ymin xmax ymax))))))))

(defun lineart-map-point (x y scale xoff yoff height rotate xmin ymin xmax ymax)
  (let* ((xf (float x))
         (yf (float y))
         (bw (max 1.0 (- xmax xmin)))
         (bh (max 1.0 (- ymax ymin)))
         (xf1 (if rotate (- yf ymin) (- xf xmin)))
         (yf1 (if rotate (- xf xmin) (- yf ymin)))
         (yf2 (- (if rotate bw bh) yf1))
         (xr (+ *lineart-x-min* xoff (* scale xf1)))
         (yr (+ *lineart-y-min* yoff (* scale yf2)))
         (xc (lineart-clamp xr *lineart-x-min* *lineart-x-max*))
         (yc (lineart-clamp yr *lineart-y-min* *lineart-y-max*))
         (z (+ *lineart-z-min*
               (* (/ (- xc *lineart-x-min*)
                     (- *lineart-x-max* *lineart-x-min*))
                  (- *lineart-z-max* *lineart-z-min*))))
         (roll (+ *lineart-roll-max*
                  (* (/ (- yc *lineart-y-max*)
                        (- *lineart-y-min* *lineart-y-max*))
                     (- *lineart-roll-min* *lineart-roll-max*)))))
    (list xc yc
          (lineart-clamp z *lineart-z-min* *lineart-z-max*)
          (lineart-clamp roll *lineart-roll-min* *lineart-roll-max*))))

(defun lineart-run (polylines width height)
  (let* ((s (lineart-compute-scale width height polylines))
         (scale (nth 0 s))
         (xoff (nth 1 s))
         (yoff (nth 2 s))
         (rotate (nth 3 s))
         (xmin (nth 4 s))
         (ymin (nth 5 s))
         (xmax (nth 6 s))
         (ymax (nth 7 s)))
    (send *ri* :servo-on)
    (when *lineart-use-joint*
      (format t "~%[LINEART-JOINT] servo-on done.~%")
      (finish-output))
    (lineart-safe-cancel-and-wait)
    (when *lineart-use-joint*
      (format t "~%[LINEART-JOINT] cancel/wait done.~%")
      (finish-output))
    (send *jedy* :angle-vector *lineart-home-av*)
    (lineart-send-av-and-wait *lineart-msec*)
    (when *lineart-use-joint*
      (format t "~%[LINEART-JOINT] home pose done.~%")
      (finish-output))
    (when *lineart-use-joint*
      (ensure-pen-tables))
    (let ((pli 0))
    (dolist (pl polylines)
      (when *lineart-use-joint*
        (format t "~%[LINEART-JOINT] polyline ~A start (points=~A).~%"
                (+ pli 1) (length pl))
        (finish-output))
      (if *lineart-use-joint*
          (let* ((mapped-pts '()))
            (dolist (pt pl)
              (let* ((mapped (lineart-map-point (car pt) (cadr pt) scale xoff yoff height rotate xmin ymin xmax ymax))
                     (x (nth 0 mapped))
                     (y (nth 1 mapped)))
                (setq mapped-pts (append mapped-pts (list (list x y))))))
            (when (eq *lineart-joint-mode* 'snap)
              (setq mapped-pts (lineart-expand-with-grid mapped-pts *pengnd-table*)))
            (let ((first t))
              (dolist (pt mapped-pts)
                (let* ((x (car pt))
                       (y (cadr pt))
                       (target (if (eq *lineart-joint-mode* 'snap)
                                   (lineart-snap-or-interp *pengnd-table* x y)
                                   (list 'interp x y)))
                       (tx (nth 1 target))
                       (ty (nth 2 target)))
                  (when first
                    (let* ((air-target (lineart-snap-or-interp *penair-table* x y t))
                           (ax (nth 1 air-target))
                           (ay (nth 2 air-target)))
                      (lineart-apply-angle (table-angle-at *penair-table* ax ay)))
                    (lineart-apply-angle (table-angle-at *pengnd-table* tx ty))
                    (setq first nil))
                  (lineart-apply-angle (table-angle-at *pengnd-table* tx ty))))))
          (let ((first t))
            (dolist (pt pl)
              (let* ((mapped (lineart-map-point (car pt) (cadr pt) scale xoff yoff height rotate xmin ymin xmax ymax))
                     (x (nth 0 mapped))
                     (y (nth 1 mapped))
                     (z (nth 2 mapped))
                     (roll (nth 3 mapped)))
                (when first
                  (setq first nil))
                (lineart-apply-point x y z roll)))))
      (send *jedy* :angle-vector *lineart-home-av*)
      (lineart-send-av-and-wait *lineart-msec*)
      (when *lineart-use-joint*
        (format t "~%[LINEART-JOINT] polyline ~A done.~%" (+ pli 1))
        (finish-output))
      (setq pli (+ pli 1))))
    (format t "~%[LINEART] sequence done.~%")))

(defun lineart-cb (msg)
  (let* ((xy (send msg :xy))
         (width (send msg :width))
         (height (send msg :height))
         (counts (send msg :counts))
         (polylines (lineart-xy-to-polylines xy counts)))
    (format t "~%[LINEART] 受信: polylines=~A points=~A~%"
            (length polylines) (truncate (/ (length xy) 2)))
    (lineart-run polylines width height)))

(defun lineart-joint-cb (msg)
  (let* ((xy (send msg :xy))
         (width (send msg :width))
         (height (send msg :height))
         (counts (send msg :counts))
         (polylines (lineart-xy-to-polylines xy counts))
         (prev *lineart-use-joint*))
    (format t "~%[LINEART-JOINT] 受信: polylines=~A points=~A~%"
            (length polylines) (truncate (/ (length xy) 2)))
    (finish-output)
    (when *lineart-joint-busy*
      (format t "~%[LINEART-JOINT] 実行中のため新しい指令をキューに積みます。~%")
      (finish-output))
    (format t "~%[LINEART-JOINT] penair/pengnd 読み込み開始...~%")
    (finish-output)
    (unless (ensure-pen-tables)
      (format t "~%[LINEART-JOINT] penair/pengnd テーブルの読み込みに失敗しました。~%")
      (finish-output)
      (return-from lineart-joint-cb nil))
    (format t "~%[LINEART-JOINT] penair/pengnd 読み込み完了。~%")
    (finish-output)
    (setq *lineart-joint-task* (list polylines width height 'interp))
    t))

(defun lineart-joint-snap-cb (msg)
  (let* ((xy (send msg :xy))
         (width (send msg :width))
         (height (send msg :height))
         (counts (send msg :counts))
         (polylines (lineart-xy-to-polylines xy counts)))
    (format t "~%[LINEART-JOINT] 受信: polylines=~A points=~A~%"
            (length polylines) (truncate (/ (length xy) 2)))
    (finish-output)
    (when *lineart-joint-busy*
      (format t "~%[LINEART-JOINT] 実行中のため新しい指令をキューに積みます。~%")
      (finish-output))
    (format t "~%[LINEART-JOINT] penair/npengnd 読み込み開始...~%")
    (finish-output)
    (unless (ensure-pen-tables)
      (format t "~%[LINEART-JOINT] penair/npengnd テーブルの読み込みに失敗しました。~%")
      (finish-output)
      (return-from lineart-joint-snap-cb nil))
    (format t "~%[LINEART-JOINT] penair/npengnd 読み込み完了。~%")
    (finish-output)
    (setq *lineart-joint-task* (list polylines width height 'snap))
    t))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 初期化関数
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun haiku-init ()
  "haiku_msgs/HaikuResult を subscribe する。
   (jedy-init) 実行後に呼び出すことを前提とする。"
  ;; *jedy* のチェック（必須ではないけど、親切な警告）
  (unless (boundp '*jedy*)
    (format t "[HAIKU] *jedy* が未定義です。先に (jedy-init) を実行してください。~%")
    (return-from haiku-init nil))
  (unless *jedy*
    (format t "[HAIKU] *jedy* が NIL です。jedy の初期化を確認してください。~%")
    (return-from haiku-init nil))

  ;; すでに subscribe 済みなら再利用
  (when *haiku-subscriber*
    (format t "[HAIKU] すでに /haiku_result を subscribe 済みです。~%")
    (return-from haiku-init *haiku-subscriber*))

  ;; ここで subscribe を張る
  (setq *haiku-subscriber*
        (ros::subscribe "haiku_result"
                        haiku_msgs::HaikuResult
                        #'haiku-result-cb
                        1)) ;; queue size=1

  (format t "[HAIKU] /haiku_result を subscribe しました。~%")
  *haiku-subscriber*)


  ;; コールバックを回すための簡単なスピン関数
(defun haiku-spin ()
  "ROSコールバックを処理するループ。
   Ctrl+C で roseus ごと止める前提。"
  (while (ros::ok)
    (when (and *lineart-joint-task* (not *lineart-joint-busy*))
      (lineart-joint-run-task))
    (ros::spin-once)
    (unix::usleep 100000))) ;; 0.1秒スリープ

(defun lineart-init ()
  "haiku_msgs/LineArt を subscribe する。"
  (unless (boundp '*jedy*)
    (format t "[LINEART] *jedy* が未定義です。先に (jedy-init) を実行してください。~%")
    (return-from lineart-init nil))
  (unless *jedy*
    (format t "[LINEART] *jedy* が NIL です。jedy の初期化を確認してください。~%")
    (return-from lineart-init nil))

  (unless *lineart-subscriber*
    (setq *lineart-subscriber*
          (ros::subscribe "lineart"
                          haiku_msgs::LineArt
                          #'lineart-cb
                          1))
    (format t "[LINEART] /lineart を subscribe しました。~%"))

  (unless *lineart-joint-subscriber*
    (setq *lineart-joint-subscriber*
          (ros::subscribe "lineart_joint"
                          haiku_msgs::LineArt
                          #'lineart-joint-cb
                          1))
    (format t "[LINEART] /lineart_joint を subscribe しました。~%"))

  (unless *lineart-joint-snap-subscriber*
    (setq *lineart-joint-snap-subscriber*
          (ros::subscribe "lineart_joint_snap"
                          haiku_msgs::LineArt
                          #'lineart-joint-snap-cb
                          1))
    (format t "[LINEART] /lineart_joint_snap を subscribe しました。~%"))

  (when (and *lineart-subscriber* *lineart-joint-subscriber*)
    (format t "[LINEART] すでに /lineart と /lineart_joint を subscribe 済みです。~%"))
  *lineart-subscriber*)
