public void NULL_DEREFERENCE(){
            for (w = 0; w <= W; w++) {
                if (i == 0 || w == 0)
                    rv[i][w] = 0;
                else if (wt[i - 1] <= w)
                    rv[i][w] = Math.max(val[i - 1] + rv[i - 1][w - wt[i - 1]], rv[i - 1][w]);

}