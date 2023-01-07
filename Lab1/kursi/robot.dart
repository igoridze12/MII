import 'dart:io';
 void main(List<String> args) {
  int up = 1;
  int left = 2;
  int down = 3;
  int right = 4;
  int hodi = 0;
  int y =0;
  if(y==0){
    if(hodi ==0){
      y= y+right+right+up+up;
    } 
    else if(hodi ==1){
      y=y+right+up+right+right;
    }
  } 
  if(y>10){
    print('poshel nahui');
  }
  else if(y<=10){
    print('win');
  }
  
}