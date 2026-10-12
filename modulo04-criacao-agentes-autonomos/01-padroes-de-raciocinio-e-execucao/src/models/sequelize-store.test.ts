import assert from "node:assert/strict";
import test from "node:test";
import { Sequelize } from "sequelize";
import { SequelizeOpsStore } from "./sequelize-store.js";

test("Sequelize adapter defines MySQL service, alert, and incident models without connecting", async () => {
  const sequelize = new Sequelize("mysql://test:test@127.0.0.1:3306/ops_pilot", {
    logging: false,
  });
  try {
    new SequelizeOpsStore(sequelize);
    assert.deepEqual(
      sequelize.modelManager.models.map((model) => model.getTableName()),
      ["services", "alerts", "incidents"],
    );
  } finally {
    await sequelize.close();
  }
});
