DIM = 256 
temp = FLTARR(DIM,DIM)

fname = 'temp.dat'
OPENR, 1, fname
READU, 1, temp
CLOSE, 1

PRINT, MIN(temp), MAX(temp)

IM = IMAGE(temp)
;!p.position=[0.1,0.1,0.9,0.9]
;contour, temp, $
;         min_value=min(temp), max_value=max(temp), $
;         /fill, xstyle=1, ystyle=1

END
