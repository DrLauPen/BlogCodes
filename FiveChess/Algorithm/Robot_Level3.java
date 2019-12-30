package Algorithm;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Robot_Level3 extends Robot_Level2 {
	protected int[][] board;
	protected Vector<String> Judge;
	protected Map<String,Integer> scoremap;
	protected float MIN = -1000000;
	protected float MAX = 1000000;
	protected Vector<Integer> points;//最可能选择的点
	protected int MaxDepth=2;
	protected int range=1;
	Robot_Level3(int color) {
		super(color);//已经设置好对应的分数权重了
		points = new Vector<Integer>();
		scoremap = new HashMap<String,Integer>();//设置好分数字典,如果搜索对方的回合,可以全部取反
		scoremap.put("长连", 100000);
		scoremap.put("活四", 10000);
		scoremap.put("冲四",2000);
		scoremap.put("活三", 1000);
		scoremap.put("冲三", 500);
		scoremap.put("活二", 100);
		scoremap.put("冲二", 30);
		scoremap.put("死四", -5);
		scoremap.put("死三", -5);
		scoremap.put("死二", -5);
	}
	@Override
	public void Generate_Next_Step() {
		next_vector = new int[2];
		float maxnum = MIN;
		if (points.size()==0) {
			/*如果暂时没有可以下的点*/
			next_vector[0] = 7;
			next_vector[1] = 7;
			return;
		}
		for(int k=0;k<points.size()-1;k+=2) {
			int i=points.get(k);
			int j=points.get(k+1);
			if (board[i][j]!=0) {
				continue;
			}
			board[i][j] = super.color;
			Vector<Integer> newpoints = Found_Points();
			float tems = Alpha_Beta(1,String.valueOf(super.anothercolor),String.valueOf(color),newpoints,MIN,MAX);
			if(tems>=maxnum) {
				maxnum = tems;
				next_vector[0] = i;
				next_vector[1] = j;
			}
			board[i][j] = 0;
		}
//		System.out.println();
	}
	@Override
	public void Get_SelfSteps(int self_i,int self_j) {
		/*设置对应的地方值*/
		if(self_i!=-1 && self_j!=-1) {
			board[self_i][self_j] = anothercolor;/*设置该点的分数为固定的tag确保不会重复下*/
			points = Found_Points();
		}
	}
	public Vector<Integer> Found_Points() {
		Vector<Integer> Points = new Vector<Integer>();
		for(int i=0;i<BORDER;i++) {
			for(int j=0;j<BORDER;j++) {
				if(board[i][j]!=0) {
					Points.addAll(Set_ThePoints(i,j));
				}
			}
		}
		return Points;
	}
	public Vector<Integer> Set_ThePoints(int posi,int posj) {
		/*重设最有可能的一些点,取已经下过的点的周围格数为4的区域点*/
		Vector<Integer> newpoints = new Vector<Integer>();//保存最有可能的点阵
		/*获取线从上往下*/
		for(int i=posi-range;i<=posi+range;i++) {
			if (i<0 || i>=super.BORDER || posj<0 || posj>=super.BORDER) {
				continue;
			}
			if(board[i][posj]==0) {
				newpoints.add(i);
				newpoints.add(posj);
			}
		}
		/*获取线从左到右*/
		for(int j=posj-range;j<=posj+range;j++) {
			if (posi<0 || posi>=super.BORDER || j<0 || j>=super.BORDER) {
				continue;
			}
			if(board[posi][j]==0) {
				newpoints.add(posi);
				newpoints.add(j);
			}
		}
		/*获取正对角线*/
		int temi = posi;
		int temj = posj;
		while(temi<=posi+range && temi>=posi-range && temj>=posj-range && temj<=posj+range) {
			if (temi<0 || temi>=super.BORDER || temj<0 || temj>=super.BORDER) {
				break;
			}
			if(board[temi][temj]==0) {
				newpoints.add(temi);
				newpoints.add(temj);
			}
			temi+=1;
			temj+=1;
		}
		temi = posi-1;
		temj = posj-1;
		while(temi<=posi+range && temi>=posi-range && temj>=posj-range && temj<=posj+range) {
			if (temi<0 || temi>=super.BORDER || temj<0 || temj>=super.BORDER) {
				break;
			}
			if(board[temi][temj]==0) {
				newpoints.add(temi);
				newpoints.add(temj);
			}
			temi-=1;
			temj-=1;
		}
		/*获取负对角线*/
		temi = posi;
		temj = posj;
		while(temi<=posi+range && temi>=posi-range && temj>=posj-range && temj<=posj+range) {
			if (temi<0 || temi>=super.BORDER || temj<0 || temj>=super.BORDER) {
				break;
			}
			if(board[temi][temj]==0) {
				newpoints.add(temi);
				newpoints.add(temj);
			}
			temi-=1;
			temj+=1;
		}
		temi = posi+1;
		temj = posj-1;
		while(temi<=posi+range && temi>=posi-range && temj>=posj-range && temj<=posj+range) {
			if (temi<0 || temi>=super.BORDER || temj<0 || temj>=super.BORDER) {
				break;
			}		
			if(board[temi][temj]==0) {
				newpoints.add(temi);
				newpoints.add(temj);
			}
			temi+=1;
			temj-=1;
		}
		return newpoints;
	}
	public float Alpha_Beta(int depth,String color,String anothercolor,Vector<Integer> points,float alpha,float beta) {
		/*进行alpha和beta剪枝以及递归的过程*/
		if (depth == MaxDepth) {
			/*如果到达了递归深度,直接返回当前局面的值就好了*/
			return Counting_Points(color,anothercolor)-2*Counting_Points(anothercolor,color);
		}
		if(depth%2 == 0) {
			for(int k=0;k<points.size()-1;k+=2) {
				int i=points.get(k);
				int j=points.get(k+1);
				board[i][j] = super.color;				
				Vector<Integer> newpoints = Found_Points();
				float s = Alpha_Beta(depth+1,anothercolor, color, newpoints,alpha,beta);
				alpha = Math.max(s, alpha);
				board[i][j] = 0;
				if(alpha>=beta) {
					/*如果该节点的下确界大于父节点的上确界,则直接剪枝*/
					return alpha;
				}
			}
			return alpha;
		}else {
			for(int k=0;k<points.size()-1;k+=2) {
				int i=points.get(k);
				int j=points.get(k+1);
				board[i][j] = super.anothercolor;
				Vector<Integer> newpoints = Found_Points();
				float s =  Alpha_Beta(depth+1,anothercolor, color, newpoints,alpha,beta);
				beta = Math.min(s, beta);
				board[i][j] = 0;
				if (alpha>=beta) {
					/*如果节点的上确界小于父节点的下确界,则直接剪枝*/
					return beta;
				}
			}
			return beta;
		}
	}
	public float Counting_Points(String color,String anothercolor) {
		/*计算当前步,整个棋盘的分数*/
		Judge = new Vector<String>();//每次清空原有内存
		for (int i=0;i<BORDER;i++) {
			Get_Columns(0,i);
			Get_Rows(i,0);
			Get_PositiveLine(i,BORDER-i);
			Get_NegativeLine(i,i);
		}	
		for(int i=0;i<BORDER-1;i++) {
			Get_PositiveLine(BORDER-1-i,i);
			Get_NegativeLine(i+1,i);
		}
		float sum = 0;
		for(int i=0;i<Judge.size();i++) {
			/*计算长连的步是否存在*/
			if (Judge_LongLink(Judge.get(i),color)) {
				sum += scoremap.get("长连");
			}
			/*判断活四*/
			if (Judge_LiveFour(Judge.get(i),color)) {
					sum += scoremap.get("活四");
			}
			/*判断冲四*/
			if (Judge_RushFour(Judge.get(i),color,anothercolor)) {
					sum += scoremap.get("冲四");
			}
			/*判断死四*/
			if (Judge_DeadFour(Judge.get(i),color,anothercolor)) {
					sum += scoremap.get("死四");				
			}
			/*判断活三*/
			if (Judge_RushThree(Judge.get(i),color,anothercolor)) {
				sum += scoremap.get("活三");
			}
			/*判断冲三*/
			if (Judge_RushThree(Judge.get(i),color,anothercolor)) {
				sum += scoremap.get("冲三");
			}
			/*判断死三*/
			if (Judge_DeadThree(Judge.get(i),color,anothercolor)) {
				sum +=scoremap.get("死三");
			}
			/*判断活二*/
			if (Judge_LiveTwo(Judge.get(i),color,anothercolor)) {
				sum +=scoremap.get("活二");
			}
			/*判断冲二*/
			if (Judge_RushTwo(Judge.get(i),color,anothercolor)) {
				sum += scoremap.get("冲二");
			}
			/*判断死二*/
			if (Judge_DeadTwo(Judge.get(i),color,anothercolor)) {
				sum += scoremap.get("死二");
			}
		}
//		System.out.print(sum+"|");
		return sum;
	}
	public void Get_Board(int[][] board) {
		/*获取对应的棋盘信息,防止深拷贝改变棋盘.*/
		this.board = new int[BORDER][BORDER];
		for(int i=0;i<BORDER;i++) {
			for(int j=0;j<BORDER;j++) {
				this.board[i][j] = board[i][j];
			}
		}
	}
	public void Get_Columns(int posi,int posj) {
		/*获取线从上往下*/
		String top = "";
		for(int i=0;i<this.BORDER;i++) {
			top = top+String.valueOf(board[i][posj]);
		}
		Judge.add(top);
	}
	public void Get_Rows(int posi,int posj) {
		/*获取线从左到右*/
		String left = "";
		for(int j=0;j<this.BORDER;j++) {
			left = left + String.valueOf(board[posi][j]);
		}
		Judge.add(left);
	}
	public void Get_PositiveLine(int posi, int posj) {
		/*获取正对角线*/
		String Pdialine = "";
		int temi = posi;
		int temj = posj;
		while(temi<this.BORDER && temi>=0 && temj>=0 && temj<this.BORDER) {
			Pdialine = Pdialine + String.valueOf(board[temi][temj]);
			temi+=1;
			temj+=1;
		}
		temi = posi-1;
		temj = posj-1;
		while(temi<this.BORDER && temi>=0 && temj>=0 && temj<this.BORDER) {
			Pdialine = String.valueOf(board[temi][temj]) + Pdialine ;
			temi-=1;
			temj-=1;
		}
		Judge.add(Pdialine);
	}	
	public void Get_NegativeLine(int posi,int posj) {
		/*获取负对角线的内容*/
		String Ndialine = "";
		int temi = posi;
		int temj = posj;
		while(temi<this.BORDER && temi>=0 && temj>=0 && temj<this.BORDER) {
			Ndialine = Ndialine + String.valueOf(board[temi][temj]);
			temi-=1;
			temj+=1;
		}
		temi = posi+1;
		temj = posj-1;
		while(temi<this.BORDER && temi>=0 && temj>=0 && temj<this.BORDER) {
			Ndialine = String.valueOf(board[temi][temj]) + Ndialine;
			temi+=1;
			temj-=1;
		}
		Judge.add(Ndialine);
	}
	public boolean Judge_LongLink(String str,String a) {
		/*判断是否存在长连*/
		Pattern pattern = Pattern.compile(a+"{5}");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_LiveFour(String str,String a) {
		/*判断是否存在活四*/
		Pattern pattern = Pattern.compile("0"+a+"{4}0");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_RushFour(String str,String a, String b) {
		/*判断是否存在冲四*/
		Pattern pattern = Pattern.compile("(^|)("+b+a+"{4}0)|(0"+a+"{4}"+b+")($|)");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_DeadFour(String str,String a, String b) {
		/*判断是否存在死四*/
		Pattern pattern = Pattern.compile("(^|)"+b+a+"{4}"+b+"($|)");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_LiveThree(String str,String a, String b) {
		/*判断是否存在活三*/
		Pattern pattern = Pattern.compile("(0"+a+"{3}0)");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_RushThree(String str,String a, String b) {
		/*判断是否存在冲三*/
		Pattern pattern = Pattern.compile("(^|)("+b+a+"{3}0)|(0"+a+"{3}"+b+")($|)");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_DeadThree(String str,String a, String b) {
		/*判断是否存在死三*/
		Pattern pattern = Pattern.compile("(^|)"+b+a+"{3}"+b+"($|)");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_LiveTwo(String str,String a, String b) {
		/*判断是否存在活二*/
		Pattern pattern = Pattern.compile("(0"+a+"{2}0)");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_RushTwo(String str,String a, String b) {
		/*判断是否存在冲二*/
		Pattern pattern = Pattern.compile("(^|)("+b+a+"{2}0)|(0"+a+"{2}"+b+")($|)");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public boolean Judge_DeadTwo(String str,String a, String b) {
		/*判断是否存在死二*/
		Pattern pattern = Pattern.compile("(^|)"+b+a+"{2}"+b+"($|)");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
}
