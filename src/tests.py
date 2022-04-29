import scipy.stats as st                                                        # knihovna pro statisticke vypocty

def t_test():
    sampleA = results[0, :, -1]
    sampleB = results[1, :, -1]
    alpha = 0.05

    t1, p1 = st.normaltest(sampleA)
    t2, p2 = st.normaltest(sampleB)

    if p1 > alpha and p2 > alpha:
        print('Vybery odpovidaji norm. rozlozeni.')
        t, p = st.ttest_ind(sampleA, sampleB)
        if (p > alpha):
            print('rozdil NENI vyznamny, rozlozeni MOHOU mit stejnou streni hodnotu, p = ', p)
        else:
            print('rozdil JE vyznamny, rozlozeni NEMOHOU mit stejnou streni hodnotu, p = ', p)
    else:
        print('Vybery neodpovidaji norm. rozlozeni.')
        t, p = st.mannwhitneyu(sampleA, sampleB)

        if (p > alpha):
            print('rozdil NENI vyznamny, stredni hodnoty si JSOU podobne, p = ', p)
        else:
            print('rozdil JE vyznamny, stredni hodnoty si NEJSOU podobne, p = ', p)
