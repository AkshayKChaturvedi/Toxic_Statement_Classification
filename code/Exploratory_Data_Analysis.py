import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import seaborn as sns

data = pd.read_csv('C:/Users/Dell/Desktop/train.csv')

# -----------------------------------------Plotting of Word Clouds Starts-----------------------------------------------

labels = list(data.columns[2:])

for label in labels:
    label_comments = data[data[label] == 1]
    word_cloud = WordCloud(background_color='white', stopwords=STOPWORDS, max_words=200, max_font_size=40,
                           random_state=2).generate(str(label_comments['comment_text']))

    print(word_cloud)
    fig = plt.figure()
    plt.imshow(word_cloud)
    plt.title(label)
    plt.axis('off')
    fig.savefig(label + "_word_cloud.png", dpi=900)

# -----------------------------------------Plotting of Word Clouds Ends-------------------------------------------------

# ----------------------------------Different categories by combination of labels Starts--------------------------------

stmt_type, count = np.unique(data.iloc[:, 2:], return_counts=True, axis=0)
freq = pd.DataFrame({'toxic': stmt_type[:, 0], 'severe_toxic': stmt_type[:, 1], 'obscene': stmt_type[:, 2],
                     'threat': stmt_type[:, 3], 'insult': stmt_type[:, 4], 'identity_hate': stmt_type[:, 5],
                     'count': count}, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate',
                                               'count'])

freq.to_csv('Different categories by combination of labels.csv')

# -----------------------------------Different categories by combination of labels Ends---------------------------------

# -----------------------------------------Statements per category or label Starts--------------------------------------

x = data.iloc[:, 2:].sum()
# plot
fig = plt.figure()
ax = sns.barplot(x.index, x.values)
plt.title("Statements per category or label")
plt.ylabel('Number of Statements', fontsize=12)
plt.xlabel('Category or Label', fontsize=12)
# adding the text labels
rects = ax.patches
counts = x.values

for rect, count in zip(rects, counts):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, count, ha='center')

fig.savefig("Statements per category or label", dpi=900)
plt.show()

# ------------------------------------------Statements per category or label Ends---------------------------------------
