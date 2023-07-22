import mmapped_df


df1 = mmapped_df.open_dataset_pl("join_data_1").lazy()
df1.sink_ipc('res.arrow')
vsdfvd
df2 = mmapped_df.open_dataset_pl("join_data_2").lazy()
df2 = df2.fetch(100)
print(df2)
df2 = df2.lazy()


join = df1.join(df2, left_on='col1', right_on='idx')
print(join.explain())
#print(join.fetch(100))
join.sink_ipc('result.arrow')
#for x in join.collect(streaming=True):
#    print(x)
