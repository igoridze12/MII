import 'dart:io';
 void main(List<String> args) {
   String box = printBox(1);
   String tree = printTree(1);
   String people = printPeople(1);
   String house = printHouse(1);
   print(box+' '+ box + ' '+ tree + ' ' + people + ' ' + house + ' ' + tree );
}
String printBox(int kolichestvo){
  String box = "|_|";
  return box;
}
String printHouse (int kolichestvo){
  String house = "|^|";
  return house;
}
String printTree(int kolichestvo){
  String tree = "*Y*";
  return tree;
}
String printPeople(int kolichestvo){
  String people = "☹";
  return people ;

}


