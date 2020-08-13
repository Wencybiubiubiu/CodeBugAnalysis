public void NULL_DEREFERENCE(){
        int[][] distance_mat = get_distance_mat(len_a,len_b);
        for (int i = 0; i < len_a; i++) {
            distance_mat[i][0] = i;
        }
        for (int j = 0; j < len_b; j++) {

}