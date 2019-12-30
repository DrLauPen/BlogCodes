package Algorithm;

import java.util.Scanner;
import java.util.Vector;

public class Receive {
	public static int[] Receive(int level,int board[][],int i,int j) {
		ChessGame game = new ChessGame();
		int[] next_vec = new int[3];
		game.Get_CheckBoard(board);/*获取棋盘信息*/
		
		//人先手
		game.self.Next_Chess(i,j);
		game.Place_Chess(i, j, game.self.Get_Color());
		game.self.Add_Chess();
		
		if(game.Judge_Winner(game.self.next_i, game.self.next_j)) {
			next_vec[2] = 1;
			return next_vec;
		};
		
		//机器输入
		switch(level) {
			case 1:
				game.enemy = new Robot_Level1(2);
				Level1(game);
				break;
			case 2:
				game.enemy = new Robot_Level2(2);
				Level2(game);
				break;
			default:
				game.enemy = new Robot_Level3(2);
				Level3(game);
		}
		if(game.Judge_Winner(game.enemy.next_i, game.enemy.next_j)){
			next_vec[2] = 2;
		}
		/*返回的位置*/
		next_vec[0] = game.enemy.next_i;
		next_vec[1] = game.enemy.next_j;
		return next_vec;
	}
	public static void Level1(ChessGame game) {
		/*等级为1的机器人的代码*/
		while(true) {
			((Robot_Level1) game.enemy).Next_Chess();
			if(game.Can_Place(game.enemy.next_i, game.enemy.next_j)) {
				break;
			}
		}
		game.Place_Chess(game.enemy.next_i, game.enemy.next_j, game.enemy.Get_Color());
		game.enemy.Add_Chess();
	}
	public static void Level2(ChessGame game) {
		/*第二等级机器人的代码*/
		while(true) {
			((Robot_Level2) game.enemy).Get_SelfSteps(game.self.next_i,game.self.next_j);
			((Robot_Level2) game.enemy).Next_Chess();
			if(game.Can_Place(game.enemy.next_i, game.enemy.next_j)) {
				break;
			}
		}
		game.Place_Chess(game.enemy.next_i, game.enemy.next_j, game.enemy.Get_Color());
		game.enemy.Add_Chess();
	}
	public static void Level3(ChessGame game) {
		/*第三等级的机器人代码*/
		while(true) {
			((Robot_Level3) game.enemy).Get_Board(game.board);//获取对应的棋盘
			((Robot_Level3) game.enemy).Get_SelfSteps(game.self.next_i,game.self.next_j);
			((Robot_Level3) game.enemy).Next_Chess();
			if(game.Can_Place(game.enemy.next_i, game.enemy.next_j)) {
				break;
			}
		}
		game.Place_Chess(game.enemy.next_i, game.enemy.next_j, game.enemy.Get_Color());
		game.enemy.Add_Chess();
	}
}
