def composite_score(s_macro, s_meso, s_micro, wM=0.5, wm=0.3, wmu=0.2):
    return wM*s_macro + wm*s_meso + wmu*s_micro
