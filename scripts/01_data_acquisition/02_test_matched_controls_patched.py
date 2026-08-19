import pandas as pd
df = pd.read_csv('matched_controls_manifest_patched.tsv', sep='\t')
matched = df[df['status'] == 'MATCHED']
print('Status:', df['status'].value_counts().to_dict())
print('Unique file_ids:', matched['file_id'].nunique())
print('Duplicate file_ids:', matched.duplicated('file_id').sum())  # must be 0
print('HCM URLs correct:', matched[matched['file_name'].str.contains('HCM', na=False)]['gcs_url'].str.contains('gdc-hcmi-open').all())
