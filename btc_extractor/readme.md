# Feature extractor for bitcoin-wallet numbers
For input list or np.array of strings outputs an np.ndarray of features.

Each feature answers the question:
   1) 24 < len < 35?
   2) starts with '1', '3' or '0'?
   3) contains only: - lowercase?
   4) - uppercase?
   5) - digits?
   6) attitude of symbols in [a..f] and [g..z]?
   7) attitude lowers/uppers?
   8) some letter (not x) repeats more than 3 times in a row?
   9) any lettering of length 4 appears more than 1 time?
  10) string has at least 1 uppercase, lowercase and digit symbols?
  11) attitude letters/digits?
  12) any chains of digits?
  13) any chains of letters?
