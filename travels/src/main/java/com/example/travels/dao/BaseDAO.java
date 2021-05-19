package com.example.travels.dao;
import org.apache.ibatis.annotations.Param;

import java.util.List;

//实现基本的CRUD功能,主要用于继承，不给定对应的方法
public interface BaseDAO<T,K>{
    void save(T t);
    void update(T t);
    void delete(K k);
    List<T> findByName();
    T findOne(K k);
    List<T> findByPages(@Param("start") Integer start, @Param("rows") Integer rows);
    //Param是Mybatis中用于对对应的参数进行对应的，其可以跟对应SQL中的参数对应
    Integer findTotals();
}
