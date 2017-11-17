import java.util.*;
import java.io.*;

public class ArabReader {
	public static void main(String[] args) {
	//C:\Users\David\IdeaProjects\ArabSeq2Seq\Data.txt
   Scanner scan = new Scanner (System.in);
	   File file = new File(scan.next());
  	 try {
     scan = new Scanner(file);
     List<String> list = new ArrayList<String>();

     scan = new Scanner(file);
     int count = 0;
     while (scan.hasNext()) {
       String s = scan.next();
       if (list.indexOf(s) == -1) { 
         list.add(s); 
       } 
       count++;
     }

     try {
      scan = new Scanner (System.in);
       PrintWriter writer = new PrintWriter(scan.next(), "UTF-8");
       scan = new Scanner(file);
       writer.println(count);
       while (scan.hasNext()) {
         String s = scan.next();
         writer.println(list.indexOf(s));
       }
       Object[] sArray = list.toArray();
       writer.println(Arrays.toString(sArray));
       writer.close();
     } catch (UnsupportedEncodingException u) {
       throw new AssertionError("File doesn't exist");
     }
   } catch (FileNotFoundException e) {
     throw new AssertionError("File doesn't exist");
   }
 }
}
