import React from "react";
import Header from "./Header";
const Shell = (props) => {
  return (
    <div>
      <Header />
      <div>{props.children}</div>
    </div>
  );
};

export default Shell;
