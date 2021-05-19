package com.example.travels.dao;

import com.example.travels.entity.User;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UserDAO {
    public void save(User user);//保存用户信息
    public User findByUsername(String username);//根据用户名查询用户
}
