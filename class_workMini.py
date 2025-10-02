scores = []
score = 0
for i in range(3):
    score = int(input("Enter score: "))
    scores.append(score)
    i+=1
print("平均得分为:", sum(scores)/len(scores))