import React from "react";
import { studios } from "./studios.json";
// import V1 from "./V1";

function StudioDetail({ match, history }) {
  console.log(studios);
  console.log(match.params.id);
  const user = studios.find((studio) => studio.id === match.params.id);
  return (
    <>
      <h2>User Detail</h2>
      <dt>id</dt>
      <dd>{user.id}</dd>
      <dt>name</dt>
      <dd>{user.name}</dd>
      <button onClick={() => history.goBack()}>Back</button>
    </>
  );
}

export default StudioDetail;
