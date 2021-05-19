package com.example.travels.service;

import com.example.travels.entity.User;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;


public interface UserService {

    void register(User user);

    User login(User user);

}
