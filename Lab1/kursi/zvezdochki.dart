import 'dart:developer';
import 'dart:io';
void main(List<String> args) {
  String k = printChelovek(2);
  String l = printDerevo(3);
  String u = printBox();
  String y = printDomik(5);
  print('8\nsad'); 
 }
 String vipolnenieCikla(int h){
  String str ='';
  int i = 0;
  while (i<h){
    str += '*';
    i++;
  }
  return str;
 }
 String printBox(){
  return vipolnenieCikla(4);
  
}
String printDomik(int h){
 return vipolnenieCikla(5);
 
}
String printDerevo(int h){
  return vipolnenieCikla(6);
}
String printChelovek(int h){
  return vipolnenieCikla(5);
}

