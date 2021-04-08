import React from "react";
import { Link } from "react-router-dom";
import { studios } from "./studios.json";

function StudioList({ match }) {
  console.log(match.url);
  //   console.log(ma);
  return (
    <>
      {/* <h2>Studio List</h2> */}
      <div>
        {studios.map(({ id, name, desc }) => (
          <div key={id}>
            <div>
              <Link to={`${match.url}/${id}`}>{name}</Link>
            </div>
            <div>{desc}</div>
          </div>
        ))}
      </div>
    </>
  );
}

export default StudioList;
