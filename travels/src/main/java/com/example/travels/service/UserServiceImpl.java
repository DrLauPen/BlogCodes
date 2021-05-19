package com.example.travels.service;

import com.example.travels.dao.UserDAO;
import com.example.travels.entity.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional
public class UserServiceImpl implements UserService{

    @Autowired
    UserDAO userDAO;

    @Override
    public void register(User user) {
        //首先查是否用相同的用户
        if(userDAO.findByUsername(user.getUsername())!=null){
            throw new RuntimeException("已经有重复用户名");
        }
        userDAO.save(user);
    }

    @Override
    public User login(User user) {
        /*用户登录*/
        User userDB = userDAO.findByUsername(user.getUsername());
        if(userDB==null){
            throw new RuntimeException("没有该用户！");
        }
        if(!userDB.getPassword().equalsIgnoreCase(user.getPassword())){
            throw new RuntimeException("密码错误");
        }
        return userDB;
    }
}
