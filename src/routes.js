import React from "react";
import "./styles.css";
import { Route, Switch, BrowserRouter as Router } from "react-router-dom";
import AppShell from "./components/navigation/AppShell";

//Start
import Concept from "./concept";
import Studios from "./studios";
import NotFound from "./components/navigation/NotFound";
import Home from "./components/navigation/Home";

import V1 from "./studios/V1";
import V2 from "./studios/V2";

const Routes = () => {
  return (
    <Router>
      <AppShell>
        <Switch>
          <Route exact path="/" component={Concept} />
          {/* <Route path="/about" component={About} /> */}

          <Route path="/studios/v1" component={V1} />
          <Route path="/studios/v2" component={V2} />

          <Route path="/studios" component={Studios} />
          <Route component={NotFound} />
        </Switch>
        {/* </main> */}
      </AppShell>
    </Router>
  );
};

export default Routes;
