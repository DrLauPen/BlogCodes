package Interface;
import java.util.Vector;

public class CManual {
	String cManual[] = { "hh hg jj ii jh gh if ig kj ij jg ji", "hh hg hj if ij ig gg gh je ii ih jg",
			"hh hg gi ig gg ii ih gh hj fj fi ei", "hh ig gi hg if gf gg jh fh hf ki fe",
			"hh ig hj hg ii jh gg jj hf ji jg gf" };
	int chess_board[][] = new int[15][15];
	Vector chessVe;

	CManual() {
		for (int i = 0; i < 15; i++)
			for (int j = 0; j < 15; j++)
				chess_board[i][j] = 0;
		chessVe= new Vector();
	}
	void choose(int flag)
	{
		String c=cManual[flag-1];
		String cut[]=null;
		cut=c.split(" ");
		int x,y;
		for(int i=0;i<cut.length;i++){
			char a[]=new char[2];
			a=cut[i].toCharArray();
			x=a[0]-96;
			y=a[1]-96;
			if(i%2==0) {
				chess_board[x-1][y-1]=2;
			}else {
				chess_board[x-1][y-1]=1;
			}
			Vector ve = new Vector();
			ve.add(x);
			ve.add(y);
			chessVe.add(ve);
		}
		//System.out.println("size->"+chessVe.size());
	}
}
