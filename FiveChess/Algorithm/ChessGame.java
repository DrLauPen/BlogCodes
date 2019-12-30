package Algorithm;

import java.util.Random;
import java.util.Scanner;
import java.util.Stack;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ChessGame {
	public int board[][];//便于直接访问的棋盘
	private int BORDER = 15; //边界长度
	public Player self;
	public Player enemy;
	ChessGame(){
		/*构造函数,创建棋盘和两个对手,白棋子默认为1,黑棋子默认为2,没有下则默认为0*/
		self = new Player(1);
	}
	public void Init_CheckBoard() {
		/*初始化棋盘,所有地方都设置为0*/
		board = new int[BORDER][BORDER];
		for(int i=0;i<BORDER;i++) {
			for(int j=0;j<BORDER;j++) {
				board[i][j] = 0;
			}
		}
	}
	public void Get_CheckBoard(int board[][]) {
		this.board = new int[BORDER][BORDER];
		for(int i=0;i<BORDER;i++) {
			for(int j=0;j<BORDER;j++) {
				this.board[i][j] = board[i][j];
			}
		}
	}
	public boolean Can_Place(int next_i,int next_j) {
		/*判断是否越界,是否已经被下过*/
		if(next_i>=BORDER | next_i<0 | next_j>=BORDER | next_j<0) {
			return false;
		}
		if(board[next_i][next_j]==0) {
			return true;
		}
		return false;
	}
	public void Place_Chess(int next_i,int next_j,int color) {
		/*放置棋子*/
		board[next_i][next_j] = color;
	}
	public boolean Judge_Winner(int lasti,int lastj) {
		/*判断胜者,算法逻辑:对每一新下的步获取其所有的对应斜对角线,正字线上所有点,如果出现五连则判断胜利.*/
		int color = board[lasti][lastj];
		/*获取线从上往下*/
		String top = "";
		for(int i=0;i<this.BORDER;i++) {
			top = top+String.valueOf(board[i][lastj]);
		}
		if (Judge_Gobang(top,color)) {
			return true;
		}
		
		/*获取线从左到右*/
		String left = "";
		for(int j=0;j<this.BORDER;j++) {
			left = left + String.valueOf(board[lasti][j]);
		}
		if (Judge_Gobang(left,color)) {
			return true;
		}
		/*获取正对角线*/
		String Pdialine = "";
		int temi = lasti;
		int temj = lastj;
		while(temi<this.BORDER && temi>=0 && temj>=0 && temj<this.BORDER) {
			Pdialine = Pdialine + String.valueOf(board[temi][temj]);
			temi+=1;
			temj+=1;
		}
		temi = lasti-1;
		temj = lastj-1;
		while(temi<this.BORDER && temi>=0 && temj>=0 && temj<this.BORDER) {
			Pdialine = String.valueOf(board[temi][temj]) + Pdialine ;
			temi-=1;
			temj-=1;
		}
		if (Judge_Gobang(Pdialine,color)) {
			return true;
		}
		String Ndialine = "";
		temi = lasti;
		temj = lastj;
		while(temi<this.BORDER && temi>=0 && temj>=0 && temj<this.BORDER) {
			Ndialine = Ndialine + String.valueOf(board[temi][temj]);
			temi-=1;
			temj+=1;
		}
		temi = lasti+1;
		temj = lastj-1;
		while(temi<this.BORDER && temi>=0 && temj>=0 && temj<this.BORDER) {
			Ndialine = String.valueOf(board[temi][temj]) + Ndialine;
			temi+=1;
			temj-=1;
		}
		if (Judge_Gobang(Ndialine,color)) {
			return true;
		}
		
		return false;
	}
	public boolean Judge_Gobang(String str,int color) {
		/*匹配是否存在五连*/
		Pattern pattern = Pattern.compile(String.valueOf(color)+"{5}");
		Matcher matcher = pattern.matcher(str);
		if(matcher.find()) {
			return true;
		}
		return false;
	}
	public void Show_CheckerBoard() {
		/*展示棋盘*/
		for(int i=0;i<BORDER;i++) {
			for(int j=0;j<BORDER;j++) {
				if(board[i][j]==2) {
					System.out.print("$ ");
				}else if(board[i][j]==1){
					System.out.print("O ");
				}else {
					System.out.print("+ ");
				}

			}
			System.out.println();
		}
	}
}
