import json
import pandas as pd

def json_to_events(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    events_list = []
    for event in data['annotations']['events']:
        events_list.append({
            'onset': event['start'],
            'duration': event['end'] - event['start'],
            'trial_type': event['label']  # e.g., 'punchline' or 'audience_laughter'
        })
    
    return pd.DataFrame(events_list)

# Example usage
events_live = json_to_events('pieman_live.json')