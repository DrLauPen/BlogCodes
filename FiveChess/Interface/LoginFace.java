package Interface;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;
import javax.swing.border.Border;

public class LoginFace implements ActionListener{
	JDialog lframe; 
	JTextField t1;
	JPasswordField t2;
	JButton b1,b2;
	JLabel l1,l2,l3;
	boolean yn;
	
	LoginFace(){
		yn = false;  //默认为账号信息不匹配
		
		lframe=new JDialog();
		lframe.setTitle("登录");
    	lframe.setSize(360,280); 
    
    	JPanel panel=new JPanel();
    	l1=new JLabel("用户名");                                
    	l2=new JLabel("密码");
    	l1.setFont(new Font("隶书",Font.BOLD,20));
    	l2.setFont(new Font("隶书",Font.BOLD,20));
    	
    	b1=new JButton("登录");
    	b2=new JButton("注册");
    	this.setButton(b1);
    	this.setButton(b2);
   	    t1=new JTextField(20);
   	    t2=new JPasswordField(20);
    	t2.setEchoChar('*');
    	
    	JLabel pa = new JLabel();
    	pa.setLayout(new GridLayout(2,2));
    	pa.add(l1);
    	pa.add(t1);
    	pa.add(l2);
    	pa.add(t2);
    	pa.setBounds(50,35, 240, 80);
    	pa.setBackground(new Color(228,243,184));
	    
    	JPanel pb = new JPanel();
    	pb.setBounds(60,155,240,40);
    	pb.setLayout(new FlowLayout());
    	pb.setBackground(new Color(228,243,184));
    	pb.add(b1);
    	pb.add(b2);
    	
	    b1.addActionListener(this);
	    b2.addActionListener(this);
	    
	    lframe.add(pa);
	    lframe.add(pb);
	    JPanel mPanel = new JPanel();
		mPanel.setSize(lframe.getSize());
		mPanel.setBackground(new Color(228,243,184));
		lframe.add(mPanel);
	    
	    lframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    	lframe.setLocationRelativeTo(null) ;  
	    lframe.setModal(true);
	    lframe.setVisible(true);  
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		// TODO 自动生成的方法存根
		if(e.getSource()==b1)
		{
			if(t1.getText().length()==0)
			{
				JOptionPane.showConfirmDialog(lframe, "用户名为空", "Logout", JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.WARNING_MESSAGE);
				return;
			}
			if(t2.getPassword().length==0)
			{
				JOptionPane.showConfirmDialog(lframe, "密码为空", "Logout", JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.WARNING_MESSAGE);
				return;
			}
		}else {
			RegisterFace ref = new RegisterFace();
			ref.rframe.dispose();
		}
	}
	
	public void setButton(JButton bu) {
		Border buttonBorder = BorderFactory.createLineBorder(Color.WHITE, 4);
		bu.setFont(new Font("隶书",Font.BOLD,20));
		bu.setBorder(buttonBorder);
		bu.setBackground(new Color(246,236,197));
	}
	
}
