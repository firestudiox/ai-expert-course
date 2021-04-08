import React from "react";
import { Route } from "react-router-dom";
import StudioList from "./StudioList";
import StudioDetail from "./StudioDetail";

const Studios = ({ match }) => {
  return (
    <>
      <h1>Studio List</h1>
      <Route exact path={match.path} component={StudioList} />
      <Route path={`${match.path}/:id`} component={StudioDetail} />
    </>
  );
};

export default Studios;
