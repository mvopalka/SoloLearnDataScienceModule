import pandas as pd
import matplotlib.pyplot as plt

presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')

plt.scatter(presidents_df['height'], presidents_df['age'],
            marker='_',
            color='m')
plt.xlabel('height')
plt.ylabel('age')
plt.title('U.S. presidents')
plt.show()
