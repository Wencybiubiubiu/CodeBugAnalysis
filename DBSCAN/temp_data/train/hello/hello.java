public void NULL_DEREFERENCE(){
  int test() {
    String s = null;
    return s.length();
  }
}

}