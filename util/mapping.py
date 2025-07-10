def mapping_pleasure_output(pleasure):
    if pleasure < -1.8:
        return 'extreme'
    elif pleasure < -0.5:
        return 'high'
    elif pleasure < -0.1:
        return 'medium'
    else:
        return 'low'

def mapping_arousal_output(arousal):
    if arousal > 1.5:
        return 'extreme'
    elif arousal > 0.5:
        return 'high'
    elif arousal > 0.01:
        return 'medium'
    else:
        return 'low'

def detect_unsatisfied(pleasure_output, arousal_output):
    if pleasure_output in ['extreme', 'high'] and arousal_output in ['extreme', 'high']:
        return 1
    elif pleasure_output in ['extreme', 'high'] and arousal_output in ['medium', 'low']:
        return 1
    else:
        return 0

