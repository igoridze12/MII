import 'dart:io';
 void main(List<String> args) {
  printLesenka();
}
 void printLesenka(){
   String str ='';
   int i = 8;
   int y= 0;
    while(i>=0){
      str += '>>>';  
    if(i%2==0){
      if (y==1){
        str +='0\n';
        y=0;
      }
      else if(y==0){
        str += '1\n';
        y=1;
      }
    }
    else if (i%2!=0){
      str += '\n';
      
    }
    
     i--;
   }
      print(str);
  }
 
