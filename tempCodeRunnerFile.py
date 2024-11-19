encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df